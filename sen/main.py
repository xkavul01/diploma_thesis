from typing import List, Dict
from datetime import datetime
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import os.path as osp
import json
import argparse

import yaml
import numpy as np
import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import RMSprop

from ocr.size_finder import find_max_resized_width
from sen.dataset import IAMDatasetSEN
from sen.model.style_extractor_network import StyleExtractorNetwork


def compute_weights(json_file: Dict) -> torch.Tensor:
    n_classes = len(json_file.keys())

    n_samples = 0
    y = []
    for key in json_file.keys():
        n_samples += len(json_file[key])
        for i in range(len(json_file[key])):
            y.append(int(key))

    return torch.from_numpy(n_samples / (n_classes * np.bincount(y))).float()


@torch.no_grad()
def evaluation(model: nn.Module,
               loss_function: nn.Module,
               loader: DataLoader,
               device: torch.device,
               epoch: int,
               n_epochs: int,
               num_classes: int,
               save_log: str
               ) -> None:
    model.eval()

    print("\nEvaluation\n")

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    running_loss = 0.0
    for i, data in enumerate(loader, start=1):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = loss_function(outputs, labels)
        running_loss += loss.item()

        probs = F.softmax(outputs, dim=-1)
        for j in range(probs.shape[0]):
            writer_id = torch.argmax(probs[j])
            confusion_matrix[labels[j]][writer_id] += 1

        if i % 10 == 0:
            true_positive = np.diag(confusion_matrix)
            false_positive = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
            false_negative = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
            true_negative = confusion_matrix.sum() - (true_positive + false_negative + false_positive)

            accuracy = ((true_positive + true_negative) / (
                        true_positive + true_negative + false_negative + false_positive + 1e-17)).mean()
            precision = (true_positive / (true_positive + false_positive + 1e-17)).mean()
            recall = (true_positive / (true_positive + false_negative + 1e-17)).mean()
            f1_score = (2 * ((precision * recall) / (precision + recall + 1e-17))).mean()
            log = f"Epoch: [{epoch + 1}/{n_epochs}]\t" \
                  f"Batch: [{i}/{len(loader.dataset) // loader.batch_size}]\t" \
                  f"cross_entropy_loss: {running_loss / 10.0}\t" \
                  f"Accuracy: {round(accuracy, 4)}\t" \
                  f"Precision: {round(precision, 4)}\t" \
                  f"Recall: {round(recall, 4)}\t" \
                  f"F1-score: {round(f1_score, 4)}"
            running_loss = 0.0
            print(log)
            with open(save_log, "a") as f:
                f.write(log + "\n")


def train(model: nn.Module,
          loss_function: nn.Module,
          loader: DataLoader,
          device: torch.device,
          optimizer: torch.optim,
          epoch: int,
          n_epochs: int,
          num_classes: int,
          save_log: str
          ) -> None:
    model.train()

    print("\nTraining\n")

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    running_loss = 0.0
    for i, data in enumerate(loader, start=1):
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = loss_function(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        probs = F.softmax(outputs.detach(), dim=-1)
        for j in range(probs.shape[0]):
            writer_id = torch.argmax(probs[j])
            confusion_matrix[labels[j]][writer_id] += 1

        if i % 10 == 0:
            true_positive = np.diag(confusion_matrix)
            false_positive = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
            false_negative = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
            true_negative = confusion_matrix.sum() - (true_positive + false_negative + false_positive)

            accuracy = ((true_positive + true_negative) / (
                        true_positive + true_negative + false_negative + false_positive + 1e-17)).mean()
            precision = (true_positive / (true_positive + false_positive + 1e-17)).mean()
            recall = (true_positive / (true_positive + false_negative + 1e-17)).mean()
            f1_score = (2 * ((precision * recall) / (precision + recall + 1e-17))).mean()
            log = f"Epoch: [{epoch + 1}/{n_epochs}]\t" \
                  f"Batch: [{i}/{len(loader.dataset) // loader.batch_size}]\t" \
                  f"cross_entropy_loss: {running_loss / 10.0}\t" \
                  f"Accuracy: {round(accuracy, 4)}\t" \
                  f"Precision: {round(precision, 4)}\t" \
                  f"Recall: {round(recall, 4)}\t" \
                  f"F1-score: {round(f1_score, 4)}"
            running_loss = 0.0
            print(log)
            with open(save_log, "a") as f:
                f.write(log + "\n")


def run(dataset: str,
        weights: str,
        device: str,
        save_dir: str,
        n_epochs: int,
        batch_size: int,
        num_workers: int,
        max_height: int,
        learning_rate: float,
        weight_decay: float,
        image_types: List[str]
        ) -> None:
    max_width, _ = find_max_resized_width(dataset, max_height, image_types)
    tmp = int(round(max_width / 256))
    final_max_width = tmp * 256 + 256 if max_width / 256 > tmp else tmp * 256

    train_split = IAMDatasetSEN(True, "train", dataset, max_height, final_max_width)
    valid_split = IAMDatasetSEN(False, "valid", dataset, max_height, final_max_width)

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

    if not osp.exists(save_dir):
        os.mkdir(save_dir)
    save_model = osp.join(save_dir, "models")
    if not osp.exists(save_model):
        os.mkdir(save_model)
    current_datetime = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    train_log = f"train_log_{current_datetime}.txt"
    eval_log = f"eval_log_{current_datetime}.txt"

    with open(osp.join(dataset, "ground_truth.json"), "r") as file:
        my_dict = json.load(file)

    torch_device = torch.device(device)
    model = StyleExtractorNetwork(len(my_dict.keys()), batch_size, False).to(torch_device)
    loss_function = nn.CrossEntropyLoss(reduction="sum", weight=compute_weights(my_dict)).to(torch_device)
    optimizer = RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    start_epoch = -1

    if weights is not None:
        if osp.exists(weights):
            checkpoint = torch.load(weights, map_location=torch.device("cpu"))
            start_epoch = checkpoint["epoch"]
            optimizer.load_state_dict(checkpoint["optimizer"])
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
              len(my_dict.keys()),
              osp.join(save_dir, train_log))

        evaluation(model,
                   loss_function,
                   valid_loader,
                   torch_device,
                   epoch,
                   n_epochs,
                   len(my_dict.keys()),
                   osp.join(save_dir, eval_log))

        checkpoint = {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "model": model.state_dict()
        }
        torch.save(checkpoint, osp.join(save_model, f"model_{epoch}.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", type=str, help="Path to the yaml config.")
    args = parser.parse_args()

    with open(args.yaml_config, "r") as f:
        config = yaml.safe_load(f)

    run(config["sen"]["dataset"],
        config["sen"]["weights"],
        config["sen"]["device"],
        config["sen"]["save_dir"],
        config["sen"]["n_epochs"],
        config["sen"]["batch_size"],
        config["sen"]["num_workers"],
        config["sen"]["max_height"],
        config["sen"]["learning_rate"],
        config["sen"]["weight_decay"],
        config["image_types"])
