from typing import Tuple, List
import os.path as osp

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
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
                 fixed_height: int,
                 fixed_width: int
                 ) -> None:
        self._image_paths, self._labels = self._load_image_paths(image_names,
                                                                 all_annotations,
                                                                 path_to_folder,
                                                                 split)
        self._alphabet = alphabet
        self._fixed_height = fixed_height
        self._fixed_width = fixed_width

        self._augmentation = augmentation
        if self._augmentation:
            self._aug_pipeline = iaa.Sequential([
                iaa.GammaContrast((0.5, 2.0), per_channel=True),
                iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.3), add=(-30, 30)),
                iaa.GaussianBlur(sigma=(1.75, 2)),
                iaa.AdditiveGaussianNoise(scale=(0.12 * 255, 0.15 * 255)),
                iaa.Rotate((-3, 3))
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

    @staticmethod
    def _centered(word_img, tsize, centering=(.5, .5), border_value=None) -> np.ndarray:
        height = tsize[0]
        width = tsize[1]

        xs, ys, xe, ye = 0, 0, width, height
        diff_h = height - word_img.shape[0]
        if diff_h >= 0:
            pv = int(centering[0] * diff_h)
            padh = (pv, diff_h - pv)
        else:
            diff_h = abs(diff_h)
            ys, ye = diff_h / 2, word_img.shape[0] - (diff_h - diff_h / 2)
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

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        image_path = self._image_paths[index]
        image = cv2.imread(image_path)

        if self._augmentation:
            new_height = int(np.random.uniform(.75, 1.25) * image.shape[0])
            new_width = int((np.random.uniform(.9, 1.1) * image.shape[1] / image.shape[0]) * new_height)
            new_height = max(4, min(self._fixed_height - 16, new_height))
            new_width = max(8, min(self._fixed_width - 32, new_width))

            image = cv2.resize(image, (new_width, new_height))
            image = self._aug_pipeline.augment_image(image)
            image = self._centered(image, (self._fixed_height, self._fixed_width), border_value=0.0)
        else:
            new_height = max(4, min(self._fixed_height - 16, image.shape[0]))
            new_width = max(8, min(self._fixed_width - 32, image.shape[1]))

            image = cv2.resize(image, (new_width, new_height))
            image = self._centered(image, (self._fixed_height, self._fixed_width), border_value=0.0)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1).float()
        image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        encoded_annotation_ctc = []
        annotation = self._labels[index]
        for i in range(len(annotation)):
            encoded_annotation_ctc.append(self._alphabet.index(annotation[i]))
        encoded_annotation_ctc = torch.Tensor(encoded_annotation_ctc)

        encoded_annotation_ce = []
        for i in range(448):
            if i < len(annotation):
                encoded_annotation_ce.append(self._alphabet.index(annotation[i]))
            elif i == len(annotation):
                encoded_annotation_ce.append(1)
            else:
                encoded_annotation_ce.append(len(self._alphabet) - 1)
        encoded_annotation_ce = torch.Tensor(encoded_annotation_ce)

        return image, encoded_annotation_ctc, encoded_annotation_ce, len(annotation)
