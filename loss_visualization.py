from typing import List
import argparse
import os.path as osp

import matplotlib.pyplot as plt


def retrieve_avg_losses_per_epoch(lines: List[str]) -> List[float]:
    avg_losses = []
    prev_epoch = -1
    tmp = []
    for line in lines:
        epoch = int(line.split()[1].split("/")[0][1:])

        if prev_epoch == -1:
            prev_epoch = epoch

        if epoch != prev_epoch:
            avg_losses.append(sum(tmp) / len(tmp))
            tmp = []
            prev_epoch = epoch

        tmp.append(float(line.split()[5]))

    return avg_losses


def retrieve_number_of_epochs(lines: List[str]) -> int:
    epochs = []
    for line in lines:
        epoch = int(line.split()[1].split("/")[0][1:])
        epochs.append(epoch)

    return len(list(set(epochs)))


def plot_ctc_loss(text_file: str, save_dir: str) -> None:
    f = open(text_file, "r")
    lines = f.readlines()
    f.close()

    list_of_losses = [float(line.split()[5]) for line in lines]
    list_of_avg_losses = retrieve_avg_losses_per_epoch(lines)
    n_epochs = retrieve_number_of_epochs(lines)

    plt.plot(range(1, len(lines) + 1), list_of_losses)
    plt.xlabel("Total number of batches")
    plt.ylabel("CTC loss")
    plt.title("CTC loss progress")
    plt.savefig(osp.join(save_dir, "ctc_loss.png"))
    plt.close()

    plt.plot(range(1, n_epochs), list_of_avg_losses)
    plt.xlabel("Number of epochs")
    plt.ylabel("CTC loss")
    plt.title("Average CTC loss per epoch")
    plt.savefig(osp.join(save_dir, "avg_loss_per_epoch.png"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_file", type=str, help="Path to the text file.")
    parser.add_argument("--save_dir", type=str, help="Path to the folder where plot will be saved.")
    args = parser.parse_args()

    plot_ctc_loss(args.text_file, args.save_dir)
