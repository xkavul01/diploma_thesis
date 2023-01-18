from typing import Tuple, List
import os.path as osp

import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
import cv2
import imgaug.augmenters as iaa


class IAMDataset(Dataset):
    def __init__(self,
                 image_names: List[str],
                 all_annotations: List[str],
                 alphabet: List[str],
                 path_to_folder: str,
                 augmentation: bool,
                 split: str,
                 max_height: int,
                 max_width: int,
                 max_length: int,
                 ) -> None:
        self._image_paths, self._labels = self._load_image_paths(image_names,
                                                                 all_annotations,
                                                                 path_to_folder,
                                                                 split)
        self._alphabet = alphabet
        self._max_height = max_height
        self._max_width = max_width
        self._max_length = max_length
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
    def _load_image_paths(image_names: List[str],
                          all_labels: List[str],
                          path_to_folder: str,
                          split: str
                          ) -> Tuple[List[str], List[str]]:
        f = open(split, "r")
        lines = f.readlines()
        lines = [line.replace("\n", "") for line in lines]
        f.close()

        image_paths = []
        labels = []
        for i in range(len(image_names)):
            form_first_part = image_names[i].split("-")[0]
            path_to_folder_2 = osp.join(path_to_folder, form_first_part)
            form = "-".join(image_names[i].split("-")[:2])
            path_to_folder_3 = osp.join(path_to_folder_2, form)

            if f"{form}-{image_names[i].split('-')[-1]}" in lines:
                image_paths.append(osp.join(path_to_folder_3, image_names[i] + ".png"))
                labels.append(all_labels[i])

        return image_paths, labels

    def __len__(self) -> int:
        return len(self._image_paths)

    def _pad_image(self, sample: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(3, self._max_height, self._max_width, dtype=torch.uint8)
        result[:, :, :sample.shape[2]] = sample

        return result

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, int]:
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

        encoded_annotation = []
        annotation = self._labels[index]
        for i in range(self._max_length):
            if i < len(annotation):
                encoded_annotation.append(self._alphabet.index(annotation[i]))
            else:
                encoded_annotation.append(len(self._alphabet) - 1)
        encoded_annotation = torch.Tensor(encoded_annotation)

        return image, encoded_annotation, len(annotation)
