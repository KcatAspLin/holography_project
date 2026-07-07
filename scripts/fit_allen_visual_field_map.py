#!/usr/bin/env python
"""Fit an affine V1-to-visual-field map from Allen-derived correspondences."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("points", help="CSV with cortical_x,cortical_y,visual_x,visual_y.")
    parser.add_argument("-o", "--out", required=True, help="Output JSON map path.")
    args = parser.parse_args()

    rows = []
    with Path(args.points).open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    cortical = np.array(
        [[float(row["cortical_x"]), float(row["cortical_y"])] for row in rows]
    )
    visual = np.array([[float(row["visual_x"]), float(row["visual_y"])] for row in rows])

    if len(cortical) < 3:
        raise ValueError("At least three correspondence points are needed.")

    design = np.concatenate([cortical, np.ones((len(cortical), 1))], axis=1)
    coef, residuals, rank, _ = np.linalg.lstsq(design, visual, rcond=None)
    if rank < 3:
        raise ValueError("Correspondence points are rank deficient.")

    matrix = coef[:2].T
    offset = coef[2]
    pred = design @ coef
    rmse = np.sqrt(np.mean((pred - visual) ** 2, axis=0))

    out = {
        "source": str(args.points),
        "input_coordinate_system": "repo V1 space coordinates",
        "output_coordinate_system": "visual-field coordinates from Allen-derived correspondences",
        "matrix": matrix.tolist(),
        "offset": offset.tolist(),
        "rmse": rmse.tolist(),
        "n_points": int(len(cortical)),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")

    print(f"Wrote {out_path}")
    print(f"RMSE: {rmse.tolist()}")


if __name__ == "__main__":
    main()
