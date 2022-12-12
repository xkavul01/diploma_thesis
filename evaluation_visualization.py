import os.path as osp
import argparse

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchaudio.models.decoder._ctc_decoder import ctc_decoder

from model.encoder import Encoder
from main import create_alphabet
from dataset import IAMDataset


@torch.no_grad()
def visualize_one_batch(dataset: str,
                        type: str,
                        cnn_model: str,
                        weights: str,
                        device: str,
                        save_dir: str,
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
    test_split = IAMDataset(image_names=image_names,
                            all_annotations=all_annotations,
                            alphabet=alphabet,
                            path_to_folder=path_to_folder,
                            augmentation=False,
                            split=osp.join(dataset, "splits", "testset.txt"),
                            max_width=max_width,
                            max_height=max_height)

    pin_memory = device == "cuda"
    test_loader = DataLoader(dataset=test_split,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    torch_device = torch.device(device)
    model = Encoder(cnn_model=cnn_model, alphabet=alphabet).to(torch_device)
    model.eval()
    decoder = ctc_decoder(lexicon=None, tokens=alphabet, blank_token="_")

    if weights != "":
        if osp.exists(weights):
            checkpoint = torch.load(weights, map_location=torch_device)
            model.load_state_dict(checkpoint["model"])
        else:
            raise ValueError("Weights do not exist.")

    images, labels, label_lengths = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)
    label_lengths = label_lengths.to(device)

    outputs = model(images)
    outputs = F.softmax(outputs, dim=2)
    outputs = outputs.permute(1, 0, 2)

    decoded_sequences = decoder(outputs.cpu())

    hypotheses = []
    ground_truths = []
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

    for idx in range(len(hypotheses)):
        print(f"Hypothesis: {hypotheses[idx]} | Ground truth: {ground_truths[idx]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_config", type=str, help="Path to the yaml config.")
    args = parser.parse_args()

    with open(args.yaml_config, "r") as f:
        config = yaml.safe_load(f)

    visualize_one_batch(config["ocr"]["dataset"],
                        config["ocr"]["type"],
                        config["ocr"]["cnn_model"],
                        config["ocr"]["weights"],
                        config["ocr"]["device"],
                        config["ocr"]["save_dir"],
                        config["ocr"]["batch_size"],
                        config["ocr"]["num_workers"],
                        config["ocr"]["max_width"],
                        config["ocr"]["max_height"])
