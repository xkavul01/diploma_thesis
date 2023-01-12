from typing import List, Tuple
import argparse
import os
import os.path as osp

import matplotlib.pyplot as plt


def retrieve_cer_and_wer(lines: List[str]) -> Tuple[List[float], List[float]]:
    cer = []
    wer = []

    for line in lines:
        if "Final" in line:
            continue

        if "CER" in line:
            cer.append(float(line.split()[3]))
            wer.append(float(line.split()[5]))

    return cer, wer


def retrieve_number_of_epochs(lines: List[str]) -> int:
    epochs = []

    for line in lines:
        if "Final" in line:
            continue

        epoch = int(line.split()[1].split("/")[0][1:])
        epochs.append(epoch)

    return len(list(set(epochs)))


def plot_cer_and_wer(text_file: str, save_dir: str) -> None:
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    f = open(text_file, "r")
    lines = f.readlines()
    f.close()

    n_epochs = retrieve_number_of_epochs(lines)
    cer_list, wer_list = retrieve_cer_and_wer(lines)

    plt.plot(range(1, n_epochs + 1), cer_list)
    plt.xlabel("Epoch")
    plt.ylabel("CER value")
    plt.title("CER values per evaluation")
    plt.savefig(osp.join(save_dir, "cer.png"))
    plt.close()

    plt.plot(range(1, n_epochs + 1), wer_list)
    plt.xlabel("Epoch")
    plt.ylabel("WER value")
    plt.title("WER values per evaluation")
    plt.savefig(osp.join(save_dir, "wer.png"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_file", type=str, help="Path to the text file.")
    parser.add_argument("--save_dir", type=str, help="Path to the folder where plot will be saved.")
    args = parser.parse_args()

    plot_cer_and_wer(args.text_file, args.save_dir)
