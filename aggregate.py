import argparse
import pathlib

import tqdm
import pandas as pd
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label", type=pathlib.Path, required=True, help="Path to the label file"
    )
    parser.add_argument(
        "--input", type=pathlib.Path, required=True, help="Path to the input file"
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        required=True,
        help="Path to the output directory",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=0.05,
        help="Radius of the neighbourhood",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of rows to process (if not specified, process all rows)",
    )
    parser.add_argument(
        "--skip-label",
        type=int,
        default=0,
        help="Label to skip (default: 0 for unlabelled points)",
    )
    parser.add_argument(
        "--saturation",
        type=float,
        default=0.1,
        help="Percentage of points from which to sample",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=30,
        help="Numer of points sampled from the query radius",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Reading input file...")
    data_df = pd.read_table(
        args.input,
        header=None,
        names=["x", "y", "z", "intensity", "r", "g", "b"],
        sep=" ",
        nrows=args.max_rows,
    )
    print("Reading label file...")
    data_df["label"] = pd.read_table(args.label, header=None, nrows=args.max_rows)
    rows_before = len(data_df)
    data_df = data_df[data_df["label"] != args.skip_label]
    print(
        f"Removed {rows_before - len(data_df)} rows with label {args.skip_label} (remaining: {len(data_df)})"
    )
    sample = data_df.sample(frac=args.saturation)
    data_matrix = data_df.to_numpy()
    radius_squared = args.radius**2
    args.output.mkdir(parents=True, exist_ok=True)

    for f in args.output.iterdir():
        f.unlink()
    print("Aggregating neighbourhoods...")
    omit_count = 0
    for idx, row in tqdm.tqdm(sample.iterrows(), total=len(sample)):
        x, y, z = row[["x", "y", "z"]].to_numpy()
        neighbourhood = data_matrix[
            (data_matrix[:, 0] - x) ** 2
            + (data_matrix[:, 1] - y) ** 2
            + (data_matrix[:, 2] - z) ** 2
            < radius_squared
        ]
        if neighbourhood.shape[0] < args.sample_size:
            omit_count += 1
            continue
        indices = np.arange(neighbourhood.shape[0])
        np.random.shuffle(indices)
        neighbourhood = neighbourhood[indices[: args.sample_size]]
        out = args.output / f"{idx}.npy"
        np.save(out, neighbourhood)
    print(f"Skipped {omit_count} query points")
    print("Done!")
