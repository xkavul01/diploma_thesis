from typing import List
import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import os.path as osp

import yaml
import cv2
import torch
import torchvision.transforms.functional as F
import imgaug.augmenters as iaa


def pad_image(sample: torch.Tensor) -> torch.Tensor:
    result = torch.zeros(3, 32, 1536, dtype=torch.uint8)
    result[:, :, :sample.shape[2]] = sample

    return result


def visualize_augmentations(path: str, save_dir: str, tr_pipeline: iaa, image_types: List[str]) -> None:
    file_list = os.listdir(path)

    for file in file_list:
        path_to_file = osp.join(path, file)

        if osp.splitext(file)[1] in image_types:
            image = cv2.imread(path_to_file)
            augmented_image = tr_pipeline.augment_image(image)
            augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
            augmented_image = torch.from_numpy(augmented_image)
            augmented_image = augmented_image.permute(2, 0, 1)
            augmented_image = pad_image(augmented_image)
            augmented_image = F.to_pil_image(augmented_image)
            augmented_image.save(osp.join(save_dir, file))

        elif osp.isdir(path_to_file):
            visualize_augmentations(path_to_file, save_dir, tr_pipeline, image_types)


def main(dataset: str, save_dir: str, image_types: List[str]) -> None:
    # edit this if you want different augmentations
    seq = iaa.Sequential([
        iaa.Resize({"height": 32, "width": "keep-aspect-ratio"}),
        iaa.GammaContrast((0.5, 2.0), per_channel=True),
        iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.3), add=(-30, 30)),
        iaa.GaussianBlur(sigma=(0, 2.5)),
        iaa.AdditiveGaussianNoise(scale=(0, 0.15 * 255)),
        iaa.ScaleX((0.8, 1.0)),
        iaa.ScaleY((0.8, 1.0)),
        iaa.ShearX((-10, 10))
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
