from collections import OrderedDict
from datetime import datetime
from typing import List, Tuple
import argparse
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import os.path as osp

import yaml
import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from jiwer import cer, wer

from dataset import IAMDataset
from size_finder import find_max_resized_width
from length_finder import find_max_length
from model.encoder import Encoder
from model.autoregressive_decoder import AutoregressiveDecoder
from model.mp_ctc import MPCTC
from model.ctc_enhanced import CTCEnhanced
from model.ocr_model import OCRModel


class SmoothCTCLoss(_Loss):
    def __init__(self, num_classes, blank=0, weight=0.01):
        super().__init__(reduction="mean")
        self.weight = weight
        self.num_classes = num_classes

        self.ctc = nn.CTCLoss(reduction="mean", blank=blank, zero_infinity=True)
        self.kldiv = nn.KLDivLoss(reduction="batchmean")

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        ctc_loss = self.ctc(log_probs, targets, input_lengths, target_lengths)

        kl_inp = log_probs.transpose(0, 1)
        kl_tar = torch.full_like(kl_inp, 1. / self.num_classes)
        kldiv_loss = self.kldiv(kl_inp, kl_tar)

        loss = (1. - self.weight) * ctc_loss + self.weight * kldiv_loss
        return loss


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    annotations = []
    images = []
    annotations_length = []

    for (i, (image, annotation, annotation_length)) in enumerate(batch):
        images.append(image.unsqueeze(0))
        annotations.append(annotation)
        annotations_length.append(int(annotation_length))

    images = torch.cat(images)
    annotations = torch.cat(annotations, dim=0)
    annotations_length = torch.IntTensor(annotations_length)

    return images, annotations, annotations_length


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


def decode_annotations(labels: torch.Tensor, label_lengths: torch.Tensor, alphabet: List[str]) -> List[str]:
    annotations = []
    count = 0
    for length in label_lengths:
        annotation = ""
        for j in range(int(length)):
            annotation = annotation + alphabet[int(labels[count + j])]
        count += int(length)
        annotations.append(annotation)

    return annotations


def preprocess_for_autoregressive_model(labels: torch.Tensor,
                                        label_lengths: torch.Tensor,
                                        alphabet: List[str],
                                        max_length: int
                                        ) -> torch.Tensor:
    target = []
    count = 0
    for length in label_lengths:
        target.append([0])
        for j in range(1, max_length + 1):
            if j < int(length):
                target[-1].append(int(labels[count + j]))
            else:
                target[-1].append(len(alphabet) - 1)
        count += length

    return torch.Tensor(target).permute(1, 0).long()


def train(model: nn.Module,
          loss_function: nn.Module,
          loader: DataLoader,
          device: torch.device,
          optimizer: torch.optim,
          epoch: int,
          n_epochs: int,
          save_log: str,
          decoder: str,
          alphabet: List[str],
          max_length: int
          ) -> None:
    model.train()
    hypotheses = []
    ground_truths = []

    print("\nTraining\n")

    for i, data in enumerate(loader, start=1):
        images, labels, label_lengths = data
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        optimizer.zero_grad()

        outputs = None
        if decoder == "base" or decoder == "mp_ctc" or decoder == "ctc_enhanced":
            outputs = model(images)
        elif decoder == "autoregressive":
            target = preprocess_for_autoregressive_model(labels, label_lengths, alphabet, max_length).to(device)
            outputs = model(images, target)

        outputs_softmax = F.softmax(outputs.detach(), dim=2)
        outputs_softmax = outputs_softmax.permute(1, 2, 0)
        outputs_decoded = greedy_decode_ctc(outputs_softmax, alphabet)
        annotations = decode_annotations(labels, label_lengths, alphabet)
        hypotheses.extend(outputs_decoded)
        ground_truths.extend(annotations)

        outputs_log_softmax = F.log_softmax(outputs, dim=2)
        output_lengths = torch.full(size=(outputs.shape[1],), fill_value=outputs.shape[0], dtype=torch.long).to(device)
        loss = loss_function(outputs_log_softmax, labels, output_lengths, label_lengths)

        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            log = f"Epoch: [{epoch + 1}/{n_epochs}]\t" \
                  f"Batch: [{i}/{len(loader.dataset) // loader.batch_size}]\t" \
                  f"ctc_loss: {loss.item()}"
            print(log)
            with open(save_log, "a") as f:
                f.write(log + "\n")

    cer_value = cer(ground_truths, hypotheses)
    wer_value = wer(ground_truths, hypotheses)
    log = f"Epoch: [{epoch + 1}/{n_epochs}]\t" \
          f"CER: {cer_value}\t" \
          f"WER: {wer_value}"
    print(log)
    with open(save_log, "a") as f:
        f.write(log + "\n\n")


@torch.no_grad()
def evaluation(model: nn.Module,
               loss_function: nn.Module,
               loader: DataLoader,
               device: torch.device,
               epoch: int,
               n_epochs: int,
               save_log: str,
               alphabet: List[str]
               ) -> None:
    model.eval()
    ground_truths = []
    hypotheses = []

    print("\nEvaluating\n")

    for i, data in enumerate(loader, start=1):
        images, labels, label_lengths = data
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        outputs = model(images)
        outputs_softmax = F.softmax(outputs.detach(), dim=2)
        outputs_softmax = outputs_softmax.permute(1, 2, 0)
        outputs_decoded = greedy_decode_ctc(outputs_softmax, alphabet)
        annotations = decode_annotations(labels, label_lengths, alphabet)
        hypotheses.extend(outputs_decoded)
        ground_truths.extend(annotations)

        outputs_log_softmax = F.log_softmax(outputs, dim=2)
        output_lengths = torch.full(size=(outputs.shape[1],), fill_value=outputs.shape[0], dtype=torch.long).to(device)
        loss = loss_function(outputs_log_softmax, labels, output_lengths, label_lengths)

        if i % 10 == 0:
            log = f"Epoch: [{epoch + 1}/{n_epochs}]\t" \
                  f"Batch: [{i}/{len(loader.dataset) // loader.batch_size}]\t" \
                  f"ctc_loss: {loss.item()}"
            print(log)
            with open(save_log, "a") as f:
                f.write(log + "\n")

    cer_value = cer(ground_truths, hypotheses)
    wer_value = wer(ground_truths, hypotheses)

    log = f"Epoch: [{epoch + 1}/{n_epochs}]\t" \
          f"CER: {cer_value}\t" \
          f"WER: {wer_value}"
    print(log)
    with open(save_log, "a") as f:
        f.write(log + "\n\n")


def init_model(decoder: str,
               cnn_model: str,
               num_layers: int,
               torch_device: torch.device,
               alphabet: List[str],
               weights_encoder: str
               ) -> OCRModel:
    if decoder == "base":
        encoder = Encoder(cnn_model, num_layers, alphabet).to(torch_device)
        return OCRModel(encoder).to(torch_device)

    elif decoder == "autoregressive":
        encoder = Encoder(cnn_model, num_layers, alphabet).to(torch_device)
        weights_encoder = torch.load(weights_encoder, map_location=torch.device("cpu"))["model"]
        weights_encoder = OrderedDict([(k.replace("_encoder.", ""), v) for k, v in weights_encoder.items()])
        encoder.load_state_dict(weights_encoder)
        decoder_model = AutoregressiveDecoder(6, cnn_model, alphabet).to(torch_device)
        return OCRModel(encoder, decoder_model).to(torch_device)

    elif decoder == "mp_ctc":
        encoder = Encoder(cnn_model, num_layers, alphabet).to(torch_device)
        weights_encoder = torch.load(weights_encoder, map_location=torch.device("cpu"))["model"]
        weights_encoder = OrderedDict([(k.replace("_encoder.", ""), v) for k, v in weights_encoder.items()])
        encoder.load_state_dict(weights_encoder)
        decoder_model = MPCTC(6, cnn_model, 0.9873, alphabet).to(torch_device)
        return OCRModel(encoder, decoder_model).to(torch_device)

    elif decoder == "ctc_enhanced":
        encoder = Encoder(cnn_model, num_layers, alphabet).to(torch_device)
        weights_encoder = torch.load(weights_encoder, map_location=torch.device("cpu"))["model"]
        weights_encoder = OrderedDict([(k.replace("_encoder.", ""), v) for k, v in weights_encoder.items()])
        encoder.load_state_dict(weights_encoder)
        decoder_model = CTCEnhanced(6, cnn_model, alphabet).to(torch_device)
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


def init_split(image_names: List[str],
               annotations: List[str],
               alphabet: List[str],
               path_to_folder: str,
               augmentation: bool,
               path_to_split: str,
               max_width: int,
               max_height: int
               ) -> IAMDataset:
    return IAMDataset(image_names,
                      annotations,
                      alphabet,
                      path_to_folder,
                      augmentation,
                      path_to_split,
                      max_height,
                      max_width)


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
        max_height: int,
        num_layers: int,
        decoder: str,
        label_smoothing: bool,
        image_types: List[str]
        ) -> None:
    f = open(osp.join(dataset, "ground_truths", type + ".txt"), "r")
    lines = f.readlines()
    lines = [line.replace("\n", "") for line in lines]
    f.close()

    image_names = [line.split()[0] for line in lines if line[0] != "#"]
    annotations, alphabet = process_annotations_and_create_alphabet(type, lines)

    path_to_folder = osp.join(dataset, type)
    max_width, _ = find_max_resized_width(path_to_folder, max_height, image_types)
    tmp = int(round(max_width / 256))
    final_max_width = tmp * 256 + 256 if max_width / 256 > tmp else tmp * 256

    path_to_train_split = osp.join(dataset, "splits", "trainset.txt")
    path_to_valid_split = osp.join(dataset, "splits", "validationset1.txt")
    train_split = init_split(image_names,
                             annotations,
                             alphabet,
                             path_to_folder,
                             True,
                             path_to_train_split,
                             final_max_width,
                             max_height)

    valid_split = init_split(image_names,
                             annotations,
                             alphabet,
                             path_to_folder,
                             False,
                             path_to_valid_split,
                             final_max_width,
                             max_height)

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
    model = init_model(decoder, cnn_model, num_layers, torch_device, alphabet, weights_encoder)

    loss_function = None
    if label_smoothing:
        loss_function = SmoothCTCLoss(len(alphabet), len(alphabet) - 1, 0.3).to(torch_device)
    else:
        loss_function = nn.CTCLoss(zero_infinity=True, blank=len(alphabet) - 1).to(torch_device)
    optimizer = Adam(model.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, 100)
    start_epoch = -1
    max_length = find_max_length(osp.join(dataset, "ground_truths", type + ".txt"))

    if weights is not None:
        if osp.exists(weights):
            checkpoint = torch.load(weights, map_location=torch.device("cpu"))
            start_epoch = checkpoint["epoch"]
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            model.load_state_dict(checkpoint["model"])
        else:
            raise ValueError("Weights do not exist.")

    for epoch in range(start_epoch + 1, n_epochs):
        train(model,
              loss_function,
              train_loader,
              torch_device,
              optimizer,
              epoch,
              n_epochs,
              osp.join(save_dir, train_log),
              decoder,
              alphabet,
              max_length)

        evaluation(model,
                   loss_function,
                   valid_loader,
                   torch_device,
                   epoch,
                   n_epochs,
                   osp.join(save_dir, eval_log),
                   alphabet)

        scheduler.step()
        checkpoint = {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "model": model.state_dict()
        }
        torch.save(checkpoint, osp.join(save_model, f"model_{epoch}.pth"))


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
        config["ocr"]["max_height"],
        config["ocr"]["num_layers"],
        config["ocr"]["decoder"],
        config["ocr"]["label_smoothing"],
        config["image_types"])
