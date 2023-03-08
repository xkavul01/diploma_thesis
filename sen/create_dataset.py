import argparse
import os
import os.path as osp
import shutil
import json
from copy import copy

import xmltodict


def create_dataset(xml_path: str, train_split: str, source_path: str, destination_path: str) -> None:
    if not osp.exists(osp.join(destination_path, "train")):
        os.mkdir(osp.join(destination_path, "train"))
    if not osp.exists(osp.join(destination_path, "valid")):
        os.mkdir(osp.join(destination_path, "valid"))
    if not osp.exists(osp.join(destination_path, "test")):
        os.mkdir(osp.join(destination_path, "test"))

    with open(train_split, "r") as file:
        tmp = file.read()
        train_set = tmp.split("\n")

    writer_dict = {}
    xml_list = os.listdir(xml_path)
    for xml_file in xml_list:
        if osp.splitext(xml_file)[1] != ".xml":
            continue

        with open(osp.join(xml_path, xml_file), "r", encoding="utf-8") as file:
            my_xml = file.read()
        my_dict = xmltodict.parse(my_xml)

        form_id = my_dict["form"]["@id"]
        first_part = form_id.split("-")[0]
        writer_id = int(my_dict["form"]["@writer-id"])

        folder_list = os.listdir(source_path)
        idx = folder_list.index(first_part)
        folder_list_2 = os.listdir(osp.join(source_path, folder_list[idx]))
        idx2 = folder_list_2.index(form_id)
        img_list = os.listdir(osp.join(source_path, folder_list[idx], folder_list_2[idx2]))

        if writer_id not in writer_dict.keys():
            writer_dict[writer_id] = []

        for img in img_list:
            if osp.splitext(img)[0] in train_set:
                writer_dict[writer_id].append(osp.join(source_path, folder_list[idx], folder_list_2[idx2], img))

    keys = list(writer_dict.keys())
    for key in keys:
        if len(writer_dict[key]) == 0:
            writer_dict.pop(key)

    new_dict = {}
    for i, key in enumerate(writer_dict.keys(), start=0):
        new_dict[i] = copy(writer_dict[key])

    for key in new_dict.keys():
        train_size = int(round(len(new_dict[key]) * 0.7))
        valid_size = int(round(len(new_dict[key]) * 0.2))

        for i in range(len(new_dict[key])):
            if i < train_size:
                shutil.copy(new_dict[key][i], osp.join(destination_path, "train"))
            elif train_size <= i < train_size + valid_size:
                shutil.copy(new_dict[key][i], osp.join(destination_path, "valid"))
            else:
                shutil.copy(new_dict[key][i], osp.join(destination_path, "test"))
            new_dict[key][i] = osp.splitext(new_dict[key][i].split(os.sep)[-1])[0]

    with open(osp.join(destination_path, "ground_truth.json"), "w") as f:
        json.dump(new_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", type=str)
    parser.add_argument("--train_split", type=str)
    parser.add_argument("--source_path", type=str)
    parser.add_argument("--destination_path", type=str)
    args = parser.parse_args()

    create_dataset(args.xml_path, args.train_split, args.source_path, args.destination_path)
