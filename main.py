from collections import OrderedDict
from datetime import datetime
from typing import List, Tuple
import argparse
import os
import os.path as osp

import yaml
import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from jiwer import cer, wer

from ocr.dataset import IAMDataset
from ocr.focal_loss import FocalLoss
from ocr.model.encoder import Encoder
from ocr.model.ctc_lstm import CTCLSTM
from ocr.model.ctc_lstm_self import CTCLSTMSELF
from ocr.model.ctc_lstm_cross import CTCLSTMCROSS
from ocr.model.ctc_enhanced import CTCEnhanced
from ocr.model.mp_ctc_lstm_self import MPCTCLSTMSELF
from ocr.model.mp_ctc_lstm_cross import MPCTCLSTMCROSS
from ocr.model.mp_ctc_lstm import MPCTCLSTM
from ocr.model.mp_ctc import MPCTC
from ocr.model.ocr_model import OCRModel


def mp_ctc_lstm_decode(output: torch.Tensor,
                       output_ctc: torch.Tensor,
                       mask_indexes: List[torch.Tensor],
                       alphabet: List[str]
                       ) -> List[str]:
    output, output_predictions = F.softmax(output, dim=-1).max(dim=-1)
    output_ctc, output_ctc_predictions = F.softmax(output_ctc, dim=-1).max(dim=-1)

    output_ctc_predictions = [torch.unique_consecutive(output_ctc_predictions[:, i])
                              for i in range(output_ctc.shape[1])]
    output_ctc_predictions = [output_ctc_predictions[i][output_ctc_predictions[i] != len(alphabet) - 1]
                              for i in range(output_ctc.shape[1])]

    for i in range(len(mask_indexes)):
        for j in range(mask_indexes[i].shape[0]):
            output_ctc_predictions[i][int(mask_indexes[i][j])] = output_predictions[int(mask_indexes[i][j]), i]

    result = []
    for i in range(len(output_ctc_predictions)):
        result.append([])
        for j in range(output_ctc_predictions[i].shape[0]):
            result[-1].append(alphabet[output_ctc_predictions[i][j]])
        result[-1] = "".join(result[-1])

    return result


"""def mask_cross_entropy(logits: torch.Tensor,
                       mask_indexes: List[torch.Tensor],
                       annotations: torch.Tensor
                       ) -> torch.Tensor:
    decoder_log_softmax = F.log_softmax(logits, dim=-1)
    result = torch.zeros(logits.shape[0])
    for i in range(result.shape[0]):
        for j in range(annotations[i].shape[0]):
            if (j == mask_indexes[i]).sum():
                result[i] -= decoder_log_softmax[i, annotations[i][j], j]

    return torch.mean(result)"""


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]) -> Tuple[torch.Tensor,
                                                                                           torch.Tensor,
                                                                                           torch.Tensor,
                                                                                           torch.Tensor]:
    annotations_ctc = []
    annotations_ce = []
    images = []
    annotations_length = []

    for (i, (image, annotation_ctc, annotation_ce, annotation_length)) in enumerate(batch):
        images.append(image.unsqueeze(0))
        annotations_ctc.extend(annotation_ctc)
        annotations_ce.append(annotation_ce.unsqueeze(0))
        annotations_length.append(int(annotation_length))

    images = torch.cat(images)
    annotations_ctc = torch.Tensor(annotations_ctc).long()
    annotations_ce = torch.cat(annotations_ce, dim=0).long()
    annotations_length = torch.IntTensor(annotations_length)

    return images, annotations_ctc, annotations_ce, annotations_length


# scores_probs should be N,C,T, blank is last class
def greedy_decode_ctc(scores_probs: torch.Tensor, chars: List[str]) -> List[str]:
    if len(scores_probs.shape) == 2:
        scores_probs = torch.cat((scores_probs[:, 0:1], scores_probs), axis=1)
        scores_probs[:, 0] = -1000
        scores_probs[-1, 1] = 1000
    else:
        scores_probs = torch.cat((scores_probs[:, :, 0:1], scores_probs), axis=2)
        scores_probs[:, :, 0] = -1000
        scores_probs[:, -1, 0] = 1000

    best = torch.argmax(scores_probs, 1) + 1
    mask = best[:, :-1] == best[:, 1:]
    best = best[:, 1:]
    best[mask] = 0
    best[best == scores_probs.shape[1]] = 0
    best = best.cpu().numpy() - 1

    outputs = []
    for line in best:
        line = line[np.nonzero(line >= 0)]
        outputs.append(''.join([chars[c] for c in line]))
    return outputs


def decode_annotations(labels: torch.Tensor,
                       label_lengths: torch.Tensor,
                       alphabet: List[str],
                       ) -> List[str]:
    annotations = []
    count = 0
    for length in label_lengths:
        annotation = ""
        for j in range(int(length)):
            annotation = annotation + alphabet[int(labels[count + j])]
        count += int(length)
        annotations.append(annotation)

    return annotations


def decode_sentences(sentences: torch.Tensor, alphabet: List[str]) -> List[str]:
    _, predictions = torch.max(sentences, dim=-1)

    decoded_sentences = []
    for i in range(predictions.shape[1]):
        decoded_sentence = ""
        for j in range(predictions.shape[0]):
            if alphabet[predictions[j][i]] == "<EOS>":
                break

            if alphabet[predictions[j][i]] == "<BLANK>":
                continue

            if (j > 0 and predictions[j][i] != predictions[j - 1][i]) or j == 0:
                decoded_sentence = decoded_sentence + alphabet[predictions[j][i]]
        decoded_sentences.append(decoded_sentence)

    return decoded_sentences


def train(model: nn.Module,
          ctc_loss: nn.Module,
          ce_loss: nn.Module,
          loader: DataLoader,
          device: torch.device,
          optimizer: torch.optim,
          epoch: int,
          n_epochs: int,
          save_log: str,
          decoder: str,
          alphabet: List[str],
          weight: float,
          encoder_freeze: bool
          ) -> None:
    model.train()
    if decoder != "base" and encoder_freeze:
        model.encoder.eval()
        model.decoder.train()

    hypotheses = []
    ground_truths = []
    all_hypotheses = []
    all_ground_truths = []
    running_loss = 0.0

    print("\nTraining\n")

    for i, data in enumerate(loader, start=1):
        images, labels_ctc, labels_ce, label_lengths = data
        images = images.to(device)
        labels_ctc = labels_ctc.to(device)
        labels_ce = labels_ce.to(device)
        label_lengths = label_lengths.to(device)

        optimizer.zero_grad()

        outputs, outputs_shortcut, outputs_ctc, outputs_decoded = None, None, None, None
        mask_indexes = None
        if decoder == "base":
            outputs, outputs_shortcut = model(images)
            outputs_softmax = F.softmax(outputs.detach(), dim=2)
            outputs_softmax = outputs_softmax.permute(1, 2, 0)
            outputs_decoded = greedy_decode_ctc(outputs_softmax, alphabet)
        else:
            if decoder == "mp_ctc_lstm" or decoder == "mp_ctc_lstm_self" or decoder == "mp_ctc_lstm_cross" or \
                    decoder == "mp_ctc":
                outputs, outputs_ctc, mask_indexes = model(images)
                outputs_decoded = mp_ctc_lstm_decode(outputs, outputs_ctc, mask_indexes, alphabet)
            else:
                outputs, outputs_ctc = model(images)
                outputs_softmax = F.softmax(outputs.detach(), dim=2)
                outputs_decoded = decode_sentences(outputs_softmax, alphabet)

        annotations = decode_annotations(labels_ctc, label_lengths, alphabet)
        hypotheses.extend(outputs_decoded)
        ground_truths.extend(annotations)

        loss = None
        if decoder == "base":
            outputs = F.log_softmax(outputs, dim=2)
            output_lengths = torch.full(size=(outputs.shape[1],),
                                        fill_value=outputs.shape[0],
                                        dtype=torch.long).to(device)
            loss = ctc_loss(outputs, labels_ctc, output_lengths, label_lengths)

            running_loss += loss.item()

            outputs_shortcut = F.log_softmax(outputs_shortcut, dim=2)
            loss += 0.1 * ctc_loss(outputs_shortcut, labels_ctc, output_lengths, label_lengths)

        elif encoder_freeze:
            outputs_ce = outputs.permute(1, 2, 0)
            loss = ce_loss(outputs_ce, labels_ce)
            running_loss += loss.item()

        else:
            outputs_ce = outputs.permute(1, 2, 0)
            outputs_ctc = F.log_softmax(outputs_ctc, dim=2)
            output_lengths = torch.full(size=(outputs_ctc.shape[1],),
                                        fill_value=outputs_ctc.shape[0],
                                        dtype=torch.long).to(device)
            loss = weight * ce_loss(outputs_ce, labels_ce) + (1. - weight) * ctc_loss(outputs_ctc,
                                                                                      labels_ctc,
                                                                                      output_lengths,
                                                                                      label_lengths)
            running_loss += loss.item()

        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            cer_value = cer(ground_truths, hypotheses)
            wer_value = wer(ground_truths, hypotheses)
            log = f"Epoch: [{epoch + 1}/{n_epochs}]\t" \
                  f"Batch: [{i}/{len(loader.dataset) // loader.batch_size}]\t" \
                  f"Loss: {running_loss / 10.0}\t" \
                  f"CER: {cer_value}\t" \
                  f"WER: {wer_value}"
            running_loss = 0.0
            all_hypotheses.extend(hypotheses)
            all_ground_truths.extend(ground_truths)
            ground_truths = []
            hypotheses = []
            print(log)
            with open(save_log, "a") as f:
                f.write(log + "\n")

    all_ground_truths.extend(ground_truths)
    all_hypotheses.extend(hypotheses)
    cer_value = cer(all_ground_truths, all_hypotheses)
    wer_value = wer(all_ground_truths, all_hypotheses)
    log = f"CER: {cer_value}\t" \
          f"WER: {wer_value}"
    print("\n" + log + "\n")
    with open(save_log, "a") as f:
        f.write("\n" + log + "\n\n")


@torch.no_grad()
def evaluation(model: nn.Module,
               ctc_loss: nn.Module,
               ce_loss: nn.Module,
               loader: DataLoader,
               device: torch.device,
               epoch: int,
               n_epochs: int,
               save_log: str,
               alphabet: List[str],
               decoder: str,
               weight: float,
               encoder_freeze: bool
               ) -> float:
    model.eval()
    if decoder != "base" and encoder_freeze:
        model.decoder.eval()

    ground_truths = []
    hypotheses = []
    all_ground_truths = []
    all_hypotheses = []
    running_loss = 0.0

    print("\nEvaluating\n")

    for i, data in enumerate(loader, start=1):
        images, labels_ctc, labels_ce, label_lengths = data
        images = images.to(device)
        labels_ctc = labels_ctc.to(device)
        labels_ce = labels_ce.to(device)
        label_lengths = label_lengths.to(device)

        outputs, outputs_ctc, outputs_decoded = None, None, None
        mask_indexes = None
        if decoder == "base":
            outputs, _ = model(images)
            outputs_softmax = F.softmax(outputs.detach(), dim=2)
            outputs_softmax = outputs_softmax.permute(1, 2, 0)
            outputs_decoded = greedy_decode_ctc(outputs_softmax, alphabet)
        else:
            if decoder == "mp_ctc_lstm" or decoder == "mp_ctc_lstm_self" or decoder == "mp_ctc_lstm_cross" or \
                    decoder == "mp_ctc":
                outputs, outputs_ctc, mask_indexes = model(images)
                outputs_decoded = mp_ctc_lstm_decode(outputs, outputs_ctc, mask_indexes, alphabet)
            else:
                outputs, outputs_ctc = model(images)
                outputs_softmax = F.softmax(outputs.detach(), dim=2)
                outputs_decoded = decode_sentences(outputs_softmax, alphabet)

        annotations = decode_annotations(labels_ctc, label_lengths, alphabet)
        hypotheses.extend(outputs_decoded)
        ground_truths.extend(annotations)

        if decoder == "base":
            outputs_log_softmax = F.log_softmax(outputs, dim=2)
            output_lengths = torch.full(size=(outputs.shape[1],), fill_value=outputs.shape[0], dtype=torch.long).to(device)
            loss = ctc_loss(outputs_log_softmax, labels_ctc, output_lengths, label_lengths)
            running_loss += loss.item()

        elif encoder_freeze:
            outputs_ce = outputs.permute(1, 2, 0)
            loss = ce_loss(outputs_ce, labels_ce)
            running_loss += loss.item()

        else:
            outputs_ce = outputs.permute(1, 2, 0)
            outputs_ctc = F.log_softmax(outputs_ctc, dim=2)
            output_lengths = torch.full(size=(outputs_ctc.shape[1],),
                                        fill_value=outputs_ctc.shape[0],
                                        dtype=torch.long).to(device)
            loss = weight * ce_loss(outputs_ce, labels_ce) + (1. - weight) * ctc_loss(outputs_ctc,
                                                                                      labels_ctc,
                                                                                      output_lengths,
                                                                                      label_lengths)
            running_loss += loss.item()

        if i % 10 == 0:
            cer_value = cer(ground_truths, hypotheses)
            wer_value = wer(ground_truths, hypotheses)
            log = f"Epoch: [{epoch + 1}/{n_epochs}]\t" \
                  f"Batch: [{i}/{len(loader.dataset) // loader.batch_size}]\t" \
                  f"Loss: {running_loss / 10.0}\t" \
                  f"CER: {cer_value}\t" \
                  f"WER: {wer_value}"
            running_loss = 0.0
            all_hypotheses.extend(hypotheses)
            all_ground_truths.extend(ground_truths)
            ground_truths = []
            hypotheses = []
            print(log)
            with open(save_log, "a") as f:
                f.write(log + "\n")

    all_ground_truths.extend(ground_truths)
    all_hypotheses.extend(hypotheses)
    cer_value = cer(all_ground_truths, all_hypotheses)
    wer_value = wer(all_ground_truths, all_hypotheses)
    log = f"CER: {cer_value}\t" \
          f"WER: {wer_value}"
    print("\n" + log + "\n")
    with open(save_log, "a") as f:
        f.write("\n" + log + "\n\n")

    return cer_value


def init_model(decoder: str,
               cnn_model: str,
               num_layers: int,
               torch_device: torch.device,
               alphabet: List[str],
               weights_encoder: str,
               dropout: float,
               encoder_freeze: bool
               ) -> OCRModel:
    if decoder == "base":
        encoder = Encoder(cnn_model, num_layers, alphabet, dropout).to(torch_device)
        return OCRModel(encoder).to(torch_device)

    elif decoder == "ctc_lstm" or \
            decoder == "mp_ctc_lstm" or decoder == "mp_ctc_lstm_self" or decoder == "mp_ctc_lstm_cross" or \
            decoder == "ctc_lstm_self" or decoder == "ctc_lstm_cross" or decoder == "ctc_enhanced" or \
            decoder == "mp_ctc":
        if encoder_freeze:
            encoder = Encoder(cnn_model, num_layers, alphabet, dropout, decoder).requires_grad_(False).to(torch_device)
        else:
            encoder = Encoder(cnn_model, num_layers, alphabet, dropout, decoder).to(torch_device)

        if weights_encoder is not None:
            weights_encoder = torch.load(weights_encoder, map_location=torch.device("cpu"))["model"]
            weights_encoder = OrderedDict([(k.replace("_encoder.", ""), v) for k, v in weights_encoder.items()])
            encoder.load_state_dict(weights_encoder)

        decoder_model = None
        if decoder == "ctc_lstm":
            decoder_model = CTCLSTM(num_layers, dropout, alphabet).to(torch_device)
        elif decoder == "ctc_lstm_self":
            decoder_model = CTCLSTMSELF(num_layers, dropout, alphabet).to(torch_device)
        elif decoder == "ctc_lstm_cross":
            decoder_model = CTCLSTMCROSS(num_layers, dropout, alphabet).to(torch_device)
        elif decoder == "ctc_enhanced":
            decoder_model = CTCEnhanced(num_layers, dropout, alphabet).to(torch_device)
        elif decoder == "mp_ctc_lstm":
            decoder_model = MPCTCLSTM(num_layers, 0.99, dropout, alphabet).to(torch_device)
        elif decoder == "mp_ctc_lstm_self":
            decoder_model = MPCTCLSTMSELF(num_layers, 0.99, dropout, alphabet).to(torch_device)
        elif decoder == "mp_ctc_lstm_cross":
            decoder_model = MPCTCLSTMCROSS(num_layers, 0.99, dropout, alphabet).to(torch_device)
        elif decoder == "mp_ctc":
            decoder_model = MPCTC(num_layers, 0.99, dropout, alphabet).to(torch_device)

        return OCRModel(encoder, decoder_model).to(torch_device)

    else:
        raise ValueError("Decoder is not available.")


def init_loader(split: IAMDataset,
                batch_size: int,
                shuffle: bool,
                num_workers: int,
                pin_memory: bool
                ) -> DataLoader:
    return DataLoader(dataset=split,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      collate_fn=collate_fn)


def create_alphabet(all_labels: List[str]) -> List[str]:
    tmp = ""
    for label in all_labels:
        tmp = tmp + label

    alphabet = list(set(tmp))
    alphabet.sort()
    alphabet = ["<SOS>", "<EOS>"] + alphabet + ["<MASK>", "<BLANK>"]

    return alphabet


def process_annotations_and_create_alphabet(type: str, lines: List[str]) -> Tuple[List[str], List[str]]:
    annotations = [line.split()[-1] for line in lines if line[0] != "#"]

    if type == "sentences" or type == "lines":
        annotations = [annotation.replace("|", " ") for annotation in annotations]

    alphabet = create_alphabet(annotations)

    return annotations, alphabet


def run(dataset: str,
        type: str,
        cnn_model: str,
        weights: str,
        weights_encoder: str,
        device: str,
        save_dir: str,
        n_epochs: int,
        batch_size: int,
        num_workers: int,
        fixed_height: int,
        fixed_width: int,
        num_layers: int,
        decoder: str,
        learning_rate: float,
        weight_decay: float,
        dropout: float,
        weight: float,
        encoder_freeze: bool,
        mask: bool
        ) -> None:
    f = open(osp.join(dataset, "ground_truths", type + ".txt"), "r")
    lines = f.readlines()
    lines = [line.replace("\n", "") for line in lines]
    f.close()

    image_names = [line.split()[0] for line in lines if line[0] != "#"]
    annotations, alphabet = process_annotations_and_create_alphabet(type, lines)

    path_to_folder = osp.join(dataset, type)
    path_to_train_split = osp.join(dataset, "splits", "trainset.txt")
    path_to_valid_split = osp.join(dataset, "splits", "validationset1.txt")
    train_split = IAMDataset(image_names,
                             annotations,
                             alphabet,
                             path_to_folder,
                             True,
                             mask,
                             path_to_train_split,
                             fixed_height,
                             fixed_width)

    valid_split = IAMDataset(image_names,
                             annotations,
                             alphabet,
                             path_to_folder,
                             False,
                             mask,
                             path_to_valid_split,
                             fixed_height,
                             fixed_width)

    pin_memory = device == "cuda"
    train_loader = init_loader(train_split, batch_size, True, num_workers, pin_memory)
    valid_loader = init_loader(valid_split, batch_size, False, num_workers, pin_memory)

    if not osp.exists(save_dir):
        os.mkdir(save_dir)
    save_model = osp.join(save_dir, "models")
    if not osp.exists(save_model):
        os.mkdir(save_model)
    current_datetime = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    train_log = f"train_log_{current_datetime}.txt"
    eval_log = f"eval_log_{current_datetime}.txt"

    torch_device = torch.device(device)
    model = init_model(decoder, cnn_model, num_layers, torch_device, alphabet, weights_encoder, dropout, encoder_freeze)

    ctc_loss = nn.CTCLoss(zero_infinity=True, blank=len(alphabet) - 1).to(torch_device)
    ce_loss = nn.CrossEntropyLoss(label_smoothing=0.2,
                                  reduction="mean",
                                  ignore_index=len(alphabet) - 1)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, [int(.33 * n_epochs), int(.66 * n_epochs)])
    start_epoch = -1

    if weights is not None:
        if osp.exists(weights):
            checkpoint = torch.load(weights, map_location=torch.device("cpu"))
            start_epoch = checkpoint["epoch"]
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            model.load_state_dict(checkpoint["model"])
        else:
            raise ValueError("Weights do not exist.")

    min_cer = None
    for epoch in range(start_epoch + 1, n_epochs):
        train(model,
              ctc_loss,
              ce_loss,
              train_loader,
              torch_device,
              optimizer,
              epoch,
              n_epochs,
              osp.join(save_dir, train_log),
              decoder,
              alphabet,
              weight,
              encoder_freeze)

        cer = evaluation(model,
                         ctc_loss,
                         ce_loss,
                         valid_loader,
                         torch_device,
                         epoch,
                         n_epochs,
                         osp.join(save_dir, eval_log),
                         alphabet,
                         decoder,
                         weight,
                         encoder_freeze)

        scheduler.step()
        checkpoint = {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model": model.state_dict()
        }
        torch.save(checkpoint, osp.join(save_model, f"model_{epoch + 1}.pth"))

        if min_cer is None or cer < min_cer:
            min_cer = cer
            torch.save(checkpoint, osp.join(save_model, f"best_model_{epoch + 1}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", type=str, help="Path to the yaml config.")
    args = parser.parse_args()

    with open(args.yaml_config, "r") as f:
        config = yaml.safe_load(f)

    run(config["ocr"]["dataset"],
        config["ocr"]["type"],
        config["ocr"]["cnn_model"],
        config["ocr"]["weights"],
        config["ocr"]["weights_encoder"],
        config["ocr"]["device"],
        config["ocr"]["save_dir"],
        config["ocr"]["n_epochs"],
        config["ocr"]["batch_size"],
        config["ocr"]["num_workers"],
        config["ocr"]["fixed_height"],
        config["ocr"]["fixed_width"],
        config["ocr"]["num_layers"],
        config["ocr"]["decoder"],
        config["ocr"]["learning_rate"],
        config["ocr"]["weight_decay"],
        config["ocr"]["dropout"],
        config["ocr"]["weight"],
        config["ocr"]["encoder_freeze"],
        config["ocr"]["mask"])
