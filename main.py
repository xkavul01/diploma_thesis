import time
from datetime import datetime
from typing import List, Tuple
import argparse
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
import torch.optim
from torch.optim import Adam
from torchaudio.models.decoder._ctc_decoder import ctc_decoder, CTCDecoder
from jiwer import cer, wer

from dataset import IAMDataset
from model.encoder import Encoder


def create_alphabet(all_labels: List[str]) -> List[str]:
    tmp = ""
    for label in all_labels:
        tmp = tmp + label

    alphabet = list(set(tmp))
    alphabet.sort()
    alphabet = ["_"] + alphabet + ["|", "<unk>"]

    return alphabet


def train(model: nn.Module,
          loss_function: nn.Module,
          loader: DataLoader,
          device: torch.device,
          optimizer: torch.optim,
          running_loss: float,
          epoch: int,
          n_epochs: int,
          save_log: str
          ) -> float:
    model.train()

    print("\nTraining\n")

    for i, data in enumerate(loader, start=1):
        images, labels, label_lengths = data
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        outputs = F.log_softmax(outputs, dim=2)
        output_lengths = torch.full(size=(images.shape[0],), fill_value=outputs.shape[0], dtype=torch.long).to(device)
        loss = loss_function(outputs, labels, output_lengths, label_lengths)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 0:
            log = f"Epoch: [{epoch + 1}/{n_epochs}]\t" \
                  f"Batch: [{i}/{len(loader.dataset) // loader.batch_size}]\t" \
                  f"ctc_loss: {loss.item()} ({running_loss / (epoch * (len(loader.dataset) // loader.batch_size) + i)})"
            print(log)
            with open(save_log, "a") as f:
                f.write(log + "\n")

    return running_loss


@torch.no_grad()
def evaluation(model: nn.Module,
               decoder: CTCDecoder,
               loader: DataLoader,
               device: torch.device,
               epoch: int,
               n_epochs: int,
               save_log: str,
               alphabet: List[str]
               ) -> Tuple[float, float]:
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
        outputs = F.softmax(outputs, dim=2)
        outputs = outputs.permute(1, 0, 2).cpu()

        start = time.time()
        decoded_sequences = decoder(outputs)
        end = time.time()

        for j in range(len(decoded_sequences)):
            translated_sequence = ""
            decoded_sequence = decoded_sequences[j][0].tokens
            for k in range(decoded_sequence.shape[0]):
                translated_sequence = translated_sequence + alphabet[int(decoded_sequence[k])]
            hypotheses.append(translated_sequence)

        for j in range(labels.shape[0]):
            translated_label = ""
            for k in range(label_lengths[j]):
                translated_label = translated_label + alphabet[int(labels[j][k])]
            ground_truths.append(translated_label)

        if i % 10 == 0:
            elapsed = end - start
            log = f"Epoch: [{epoch + 1}/{n_epochs}]\t" \
                  f"Batch: [{i}/{len(loader.dataset) // loader.batch_size}]\t" \
                  f"Decoding time: {elapsed * 1000} ms"
            print(log)
            with open(save_log, "a") as f:
                f.write(log + "\n")

    cer_value = cer(ground_truths, hypotheses)
    wer_value = wer(ground_truths, hypotheses)
    if epoch == -1:
        log = f"Final evaluation on test split\t" \
              f"CER: {cer_value}\t" \
              f"WER: {wer_value}"
        print(log)
        with open(save_log, "a") as f:
            f.write(log + "\n")
    else:
        log = f"Epoch: [{epoch + 1}/{n_epochs}]\t" \
              f"CER: {cer_value}\t" \
              f"WER: {wer_value}"
        print(log)
        with open(save_log, "a") as f:
            f.write(log + "\n")

    return cer_value, wer_value


def run(dataset: str,
        type: str,
        cnn_model: str,
        weights: str,
        device: str,
        test: bool,
        save_dir: str,
        n_epochs: int,
        batch_size: int,
        num_workers: int,
        max_width: int,
        max_height: int
        ) -> None:
    f = open(osp.join(dataset, "ground_truths", type + ".txt"), "r")
    lines = f.readlines()
    lines = [line.replace("\n", "") for line in lines]
    f.close()

    image_names = [line.split()[0] for line in lines if line[0] != "#"]
    all_annotations = [line.split()[-1] for line in lines if line[0] != "#"]

    tmp_bool = type == "sentences" or type == "lines"
    all_annotations = [annotation.replace("|", " ") for annotation in all_annotations] if tmp_bool else all_annotations
    alphabet = create_alphabet(all_annotations)

    path_to_folder = osp.join(dataset, type)
    train_split = IAMDataset(image_names=image_names,
                             all_annotations=all_annotations,
                             alphabet=alphabet,
                             path_to_folder=path_to_folder,
                             augmentation=True,
                             split=osp.join(dataset, "splits", "trainset.txt"),
                             max_width=max_width,
                             max_height=max_height)
    valid_split = IAMDataset(image_names=image_names,
                             all_annotations=all_annotations,
                             alphabet=alphabet,
                             path_to_folder=path_to_folder,
                             augmentation=False,
                             split=osp.join(dataset, "splits", "validationset1.txt"),
                             max_width=max_width,
                             max_height=max_height)
    test_split = IAMDataset(image_names=image_names,
                            all_annotations=all_annotations,
                            alphabet=alphabet,
                            path_to_folder=path_to_folder,
                            augmentation=False,
                            split=osp.join(dataset, "splits", "testset.txt"),
                            max_width=max_width,
                            max_height=max_height)

    pin_memory = device == "cuda"
    train_loader = DataLoader(dataset=train_split,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(dataset=valid_split,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    test_loader = DataLoader(dataset=test_split,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    if not osp.exists(save_dir):
        os.mkdir(save_dir)
    save_model = osp.join(save_dir, "models")
    if not osp.exists(save_model):
        os.mkdir(save_model)
    current_datetime = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    train_log = f"train_log_{current_datetime}.txt"
    eval_log = f"eval_log_{current_datetime}.txt"

    torch_device = torch.device(device)
    model = Encoder(cnn_model=cnn_model, alphabet=alphabet).to(torch_device)
    loss_function = nn.CTCLoss().to(torch_device)
    optimizer = Adam(model.parameters(), lr=0.0001)
    decoder = ctc_decoder(lexicon=None, tokens=alphabet, blank_token="_")
    min_cer = -1
    start_epoch = -1

    if weights != "":
        if osp.exists(weights):
            checkpoint = torch.load(weights, map_location=torch.device("cpu"))
            start_epoch = checkpoint["epoch"]
            min_cer = checkpoint["min_cer"]
            optimizer.load_state_dict(checkpoint["optimizer"])
            model.load_state_dict(checkpoint["model"])
        else:
            raise ValueError("Weights do not exist.")

    if test:
        evaluation(model,
                   decoder,
                   test_loader,
                   torch_device,
                   -1,
                   -1,
                   osp.join(save_dir, eval_log),
                   alphabet)
    else:
        running_loss = 0.0

        for epoch in range(start_epoch + 1, n_epochs):
            running_loss = train(model,
                                 loss_function,
                                 train_loader,
                                 torch_device,
                                 optimizer,
                                 running_loss,
                                 epoch,
                                 n_epochs,
                                 osp.join(save_dir, train_log))
            cer_value, wer_value = evaluation(model,
                                              decoder,
                                              valid_loader,
                                              torch_device,
                                              epoch,
                                              n_epochs,
                                              osp.join(save_dir, eval_log),
                                              alphabet)
            checkpoint = {
                "epoch": epoch,
                "min_cer": min_cer,
                "optimizer": optimizer.state_dict(),
                "model": model.state_dict()
            }
            torch.save(checkpoint, osp.join(save_model, f"model_{epoch}.pth"))

            if cer_value < min_cer or min_cer == -1:
                min_cer = cer_value
                torch.save(checkpoint, osp.join(save_model, "model_best.pth"))

        best_model = torch.load(osp.join(save_model, "model_best.pth"))
        model.load_state_dict(best_model["model"])
        evaluation(model,
                   decoder,
                   test_loader,
                   torch_device,
                   -1,
                   -1,
                   osp.join(save_dir, eval_log),
                   alphabet)


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
        config["ocr"]["device"],
        config["ocr"]["test"],
        config["ocr"]["save_dir"],
        config["ocr"]["n_epochs"],
        config["ocr"]["batch_size"],
        config["ocr"]["num_workers"],
        config["ocr"]["max_width"],
        config["ocr"]["max_height"])
