from typing import List
from datetime import datetime
import os
import os.path as osp
import argparse

import yaml
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from jiwer import cer, wer

from ocr.main import greedy_decode_ctc
from ocr.main import decode_annotations
from ocr.main import process_annotations_and_create_alphabet
from ocr.main import init_loader
from ocr.main import init_model
from ocr.dataset import IAMDataset


@torch.no_grad()
def test(model: nn.Module,
         loss_function: nn.Module,
         loader: DataLoader,
         device: torch.device,
         save_log: str,
         output_log: str,
         alphabet: List[str]
         ) -> None:
    running_loss = 0.0
    ground_truths = []
    hypotheses = []

    for i, data in enumerate(loader, start=1):
        images, labels, label_lengths = data
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        outputs, _ = model(images)
        outputs_softmax = F.softmax(outputs.detach(), dim=2)
        outputs_softmax = outputs_softmax.permute(1, 2, 0)
        outputs_decoded = greedy_decode_ctc(outputs_softmax, alphabet)
        annotations = decode_annotations(labels, label_lengths, alphabet)
        hypotheses.extend(outputs_decoded)
        ground_truths.extend(annotations)

        with open(output_log, "a") as f:
            log = "=====================================================================\n"
            log += f"Batch [{i - 1}/{len(loader.dataset) // loader.batch_size}]:\n\n"
            for j in range(len(outputs_decoded)):
                log += f"Hypothesis: {outputs_decoded[j]}\nGround truth: {annotations[j]}\n\n"
            print(log)
            f.write(log + "\n")

        outputs = F.log_softmax(outputs, dim=2)
        output_lengths = torch.full(size=(outputs.shape[1],), fill_value=outputs.shape[0], dtype=torch.long).to(device)
        loss = loss_function(outputs, labels, output_lengths, label_lengths)
        running_loss += loss.item()

        if i % 10 == 0:
            cer_value = cer(ground_truths, hypotheses)
            wer_value = wer(ground_truths, hypotheses)
            log = f"Batch: [{i - 1}/{len(loader.dataset) // loader.batch_size}]\t" \
                  f"ctc_loss: {running_loss / 10.0}\t" \
                  f"CER: {cer_value}\t" \
                  f"WER: {wer_value}"
            running_loss = 0.0
            ground_truths = []
            hypotheses = []
            print(log)
            with open(save_log, "a") as f:
                f.write(log + "\n")


def run(dataset: str,
        type: str,
        cnn_model: str,
        weights: str,
        weights_encoder: str,
        device: str,
        save_dir: str,
        batch_size: int,
        num_workers: int,
        fixed_height: int,
        fixed_width: int,
        num_layers: int,
        decoder: str,
        dropout: float
        ) -> None:
    f = open(osp.join(dataset, "ground_truths", type + ".txt"), "r")
    lines = f.readlines()
    lines = [line.replace("\n", "") for line in lines]
    f.close()

    image_names = [line.split()[0] for line in lines if line[0] != "#"]
    annotations, alphabet = process_annotations_and_create_alphabet(type, lines)

    path_to_folder = osp.join(dataset, type)

    path_to_test_split = osp.join(dataset, "splits", "testset.txt")
    test_split = IAMDataset(image_names,
                            annotations,
                            alphabet,
                            path_to_folder,
                            False,
                            path_to_test_split,
                            fixed_height,
                            fixed_width)

    test_loader = init_loader(test_split,
                              batch_size,
                              False,
                              num_workers,
                              device == "cuda")

    if not osp.exists(save_dir):
        os.mkdir(save_dir)
    current_datetime = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    test_log = f"test_log_{current_datetime}.txt"
    output_log = f"output_log_{current_datetime}.txt"

    torch_device = torch.device(device)
    model = init_model(decoder, cnn_model, num_layers, torch_device, alphabet, weights_encoder, dropout)
    model.eval()
    loss_function = nn.CTCLoss(zero_infinity=True, blank=len(alphabet) - 1).to(torch_device)

    if weights is not None:
        if osp.exists(weights):
            checkpoint = torch.load(weights, map_location=torch.device("cpu"))
            model.load_state_dict(checkpoint["model"])
        else:
            raise ValueError("Weights do not exist.")

    test(model,
         loss_function,
         test_loader,
         torch_device,
         osp.join(save_dir, test_log),
         osp.join(save_dir, output_log),
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
        config["ocr"]["weights_encoder"],
        config["ocr"]["device"],
        config["ocr"]["save_dir"],
        config["ocr"]["batch_size"],
        config["ocr"]["num_workers"],
        config["ocr"]["fixed_height"],
        config["ocr"]["fixed_width"],
        config["ocr"]["num_layers"],
        config["ocr"]["decoder"],
        config["ocr"]["dropout"])