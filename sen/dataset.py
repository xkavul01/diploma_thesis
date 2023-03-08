import os
from typing import Tuple, List
import os.path as osp
import json

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import cv2
import imgaug.augmenters as iaa


class IAMDatasetSEN(Dataset):
    def __init__(self, augmentation: bool, split: str, source_path: str, max_height: int, max_width: int) -> None:
        super(IAMDatasetSEN, self).__init__()

        self._image_paths, self._labels = self._load_image_paths_and_labels(osp.join(source_path, split),
                                                                            osp.join(source_path, "ground_truth.json"))

        self._max_height = max_height
        self._max_width = max_width

        self._augmentation = augmentation
        if self._augmentation:
            self._aug_pipeline = iaa.Sequential([
                iaa.Resize({"height": self._max_height, "width": "keep-aspect-ratio"}),
                iaa.GammaContrast((0.5, 2.0), per_channel=True),
                iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.3), add=(-30, 30)),
                iaa.GaussianBlur(sigma=(0, 2.5)),
                iaa.AdditiveGaussianNoise(scale=(0, 0.15 * 255)),
                iaa.ScaleX((0.8, 1.0)),
                iaa.ScaleY((0.8, 1.0)),
                iaa.ShearX((-10, 10))
            ])

    @staticmethod
    def _load_image_paths_and_labels(split: str, json_file: str) -> Tuple[List[str], List[int]]:
        image_paths = []
        labels = []

        with open(json_file, "r") as file:
            my_dict = json.load(file)

        img_list = os.listdir(split)
        for img in img_list:
            for key in my_dict.keys():
                if osp.splitext(img)[0] in my_dict[key]:
                    image_paths.append(osp.join(split, img))
                    labels.append(int(key))

        return image_paths, labels

    def __len__(self) -> int:
        return len(self._image_paths)

    def _pad_image(self, sample: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(3, self._max_height, self._max_width, dtype=torch.uint8)
        result[:, :, :sample.shape[2]] = sample

        return result

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image_path = self._image_paths[index]
        image = cv2.imread(image_path)

        if self._augmentation:
            image = self._aug_pipeline.augment_image(image)
        else:
            image = iaa.Resize({"height": self._max_height, "width": "keep-aspect-ratio"}).augment_image(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        image = self._pad_image(image).float()
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        label = self._labels[index]

        return image, label
