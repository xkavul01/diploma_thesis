import argparse


def find_max_length(path_to_file: str) -> int:
    f = open(path_to_file, "r")
    lines = f.readlines()
    f.close()

    max_length = 0
    for line in lines:
        if line[0] != "#":
            line = line.split()[-1]
            if max_length < len(line):
                max_length = len(line)

    return max_length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Path to the text file.")
    args = parser.parse_args()

    max_length = find_max_length(args.file)
    print(max_length)


if __name__ == "__main__":
    main()
