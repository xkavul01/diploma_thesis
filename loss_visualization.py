from typing import List
import argparse
import os
import os.path as osp

import matplotlib.pyplot as plt


def retrieve_avg_losses_per_epoch(lines: List[str]) -> List[float]:
    avg_losses = []
    prev_epoch = -1
    tmp = []
    for line in lines:
        if line == "\n" or "CER" in line or "Epoch of the best model" in line:
            continue

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
        if line == "\n" or "Epoch of the best model" in line:
            continue

        epoch = int(line.split()[1].split("/")[0][1:])
        epochs.append(epoch)

    return len(list(set(epochs)))


def plot_ctc_loss(train_log: str, eval_log: str, save_dir: str) -> None:
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    f = open(train_log, "r")
    lines_train = f.readlines()
    f.close()
    f = open(eval_log, "r")
    lines_eval = f.readlines()
    f.close()

    avg_losses_train = retrieve_avg_losses_per_epoch(lines_train)
    avg_losses_eval = retrieve_avg_losses_per_epoch(lines_eval)
    n_epochs = retrieve_number_of_epochs(lines_train)

    plt.plot(range(1, n_epochs), avg_losses_train)
    plt.plot(range(1, n_epochs), avg_losses_eval)
    plt.legend(["Train loss", "Validation loss"])
    plt.xlabel("Number of epochs")
    plt.ylabel("Average CTC loss")
    plt.title("Average CTC losses per epochs")
    plt.savefig(osp.join(save_dir, "avg_loss_per_epoch.png"))
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_log", type=str, help="Path to the train log.")
    parser.add_argument("--eval_log", type=str, help="Path to the evaluation log.")
    parser.add_argument("--save_dir", type=str, help="Path to the folder where plot will be saved.")
    args = parser.parse_args()

    plot_ctc_loss(args.train_log, args.eval_log, args.save_dir)
