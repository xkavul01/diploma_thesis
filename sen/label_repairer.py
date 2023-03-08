import argparse
import json
from copy import copy
import os.path as osp


def repair_labels(root: str) -> None:
    with open(osp.join(root, "ground_truth_old.json"), "r") as file:
        old_dict = json.load(file)

    new_dict = {}
    for i, key in enumerate(old_dict.keys(), start=0):
        new_dict[i] = copy(old_dict[key])

    with open(osp.join(root, "ground_truth.json"), "w") as file:
        json.dump(new_dict, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    args = parser.parse_args()

    repair_labels(args.root)
