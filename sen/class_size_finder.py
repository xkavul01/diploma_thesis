import argparse
import json


def find_size(json_file: str) -> None:
    with open(json_file, "r") as file:
        my_dict = json.load(file)

    max_size = -1
    class_index = None
    for key in my_dict.keys():
        print(key, len(my_dict[key]))

        if max_size < len(my_dict[key]):
            max_size = len(my_dict[key])
            class_index = key

    print(f"Writer ID: {class_index} Size: {max_size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_file", type=str)
    args = parser.parse_args()

    find_size(args.json_file)
