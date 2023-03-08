from typing import List
import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import os.path as osp

import yaml
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
import imgaug.augmenters as iaa


def centered(word_img, tsize, centering=(.5, .5), border_value=None):
    height = tsize[0]
    width = tsize[1]

    xs, ys, xe, ye = 0, 0, width, height
    diff_h = height-word_img.shape[0]
    if diff_h >= 0:
        pv = int(centering[0] * diff_h)
        padh = (pv, diff_h-pv)
    else:
        diff_h = abs(diff_h)
        ys, ye = diff_h/2, word_img.shape[0] - (diff_h - diff_h/2)
        padh = (0, 0)
    diff_w = width - word_img.shape[1]
    if diff_w >= 0:
        pv = int(centering[1] * diff_w)
        padw = (pv, diff_w - pv)
    else:
        diff_w = abs(diff_w)
        xs, xe = diff_w / 2, word_img.shape[1] - (diff_w - diff_w / 2)
        padw = (0, 0)

    if border_value is None:
        border_value = np.median(word_img)
    word_img = np.pad(word_img[ys:ye, xs:xe], (padh, padw, (0, 0)), 'constant', constant_values=border_value)
    return word_img


def visualize_augmentations(path: str, save_dir: str, tr_pipeline: iaa, image_types: List[str]) -> None:
    file_list = os.listdir(path)

    for file in file_list:
        path_to_file = osp.join(path, file)

        if osp.splitext(file)[1] in image_types:
            image = cv2.imread(path_to_file)
            new_height = int(np.random.uniform(.75, 1.25) * image.shape[0])
            new_width = int((np.random.uniform(.9, 1.1) * image.shape[1] / image.shape[0]) * new_height)
            new_height, new_width = max(4, min(128 - 16, new_height)), max(8, min(1792 - 32, new_width))

            image = cv2.resize(image, (new_width, new_height))
            image = tr_pipeline.augment_image(image)
            image = centered(image, (128, 1792), border_value=0.0)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1)
            image = F.to_pil_image(image)
            image.save(osp.join(save_dir, file))

        elif osp.isdir(path_to_file):
            visualize_augmentations(path_to_file, save_dir, tr_pipeline, image_types)


def main(dataset: str, save_dir: str, image_types: List[str]) -> None:
    # edit this if you want different augmentations
    seq = iaa.Sequential([
        iaa.GammaContrast((0.5, 2.0), per_channel=True),
        iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.3), add=(-30, 30)),
        iaa.GaussianBlur(sigma=(1.75, 2)),
        iaa.AdditiveGaussianNoise(scale=(0.12 * 255, 0.15 * 255)),
        iaa.Rotate((-3, 3))
        ])

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    visualize_augmentations(dataset, save_dir, seq, image_types)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Path to the dataset we want to visualize augmentation on.")
    parser.add_argument("--save_dir", type=str, help="Path to the folder where visualizations will be saved.")
    parser.add_argument("--yaml_config", type=str, help="Path to the yaml config.")
    args = parser.parse_args()

    with open(args.yaml_config, "r") as f:
        config = yaml.safe_load(f)

    main(args.dataset, args.save_dir, config["image_types"])
