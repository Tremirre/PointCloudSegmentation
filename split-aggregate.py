import os
import random
import pathlib
import argparse
import typing
import glob


class Args(typing.NamedTuple):
    aggregate_dir: pathlib.Path
    target_dir: pathlib.Path
    val_percentage: float
    test_percentage: float
    seed: int


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregate-dir",
        type=pathlib.Path,
        required=True,
        help="Path to the aggregate directory",
    )
    parser.add_argument(
        "--target-dir",
        type=pathlib.Path,
        required=True,
        help="Path to the target directory",
    )
    parser.add_argument(
        "--val-percentage",
        type=float,
        default=0.05,
        help="Percentage of data to use for validation",
    )
    parser.add_argument(
        "--test-percentage",
        type=float,
        default=0.05,
        help="Percentage of data to use for testing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    if not os.path.exists(args.aggregate_dir):
        raise FileNotFoundError(f"{args.aggregate_dir} not found")

    data_files = glob.glob(str(args.aggregate_dir) + "/*.npy")
    n_files = len(data_files)
    n_val = int(n_files * args.val_percentage)
    n_test = int(n_files * args.test_percentage)
    print(f"Number of files: {n_files}")
    print(f"Number of validation files: {n_val}")
    print(f"Number of test files: {n_test}")
    random.shuffle(data_files)
    val_files = data_files[:n_val]
    test_files = data_files[n_val : n_val + n_test]
    train_files = data_files[n_val + n_test :]

    os.makedirs(args.aggregate_dir / "train", exist_ok=True)
    os.makedirs(args.aggregate_dir / "val", exist_ok=True)
    os.makedirs(args.aggregate_dir / "test", exist_ok=True)

    for file in val_files:
        os.rename(file, args.target_dir / "val" / pathlib.Path(file).name)

    for file in test_files:
        os.rename(file, args.target_dir / "test" / pathlib.Path(file).name)

    for file in train_files:
        os.rename(file, args.target_dir / "train" / pathlib.Path(file).name)


if __name__ == "__main__":
    main()
