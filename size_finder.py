from typing import Tuple, List
import argparse
import os
import os.path as osp

import yaml
import PIL
from PIL import Image


def find_max_resized_width(path: str, resized_height: int, image_types: List[str]) -> Tuple[int, str]:
    file_list = os.listdir(path)
    max_width = 0
    image_label = None

    for file in file_list:
        path_to_file = osp.join(path, file)

        if osp.splitext(file)[1] in image_types:
            try:
                image = Image.open(path_to_file)
                width, height = image.size
                aspect_ratio = height / width
                resized_width = int(resized_height / aspect_ratio)

                if resized_width > max_width:
                    max_width = resized_width
                    image_label = osp.splitext(file)[0]

            except PIL.UnidentifiedImageError:
                continue

        elif osp.isdir(path_to_file):
            width, tmp = find_max_resized_width(path=path_to_file,
                                                resized_height=resized_height,
                                                image_types=image_types)

            if width > max_width:
                max_width = width
                image_label = tmp

    return max_width, image_label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, help="Path to the folder where we want to find out max resized width.")
    parser.add_argument("--height", type=int, help="Fixed resized height of all images.")
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    width, image_label = find_max_resized_width(
        path=args.root,
        resized_height=args.height,
        image_types=config["size_finder"]["image_types"]
    )

    print(f"Image label: {image_label}\n"
          f"Max resized width: {width}")


if __name__ == "__main__":
    main()
