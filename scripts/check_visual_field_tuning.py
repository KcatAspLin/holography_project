#!/usr/bin/env python
"""Inspect the modified Eq. 8 visual-field orientation factor."""

import argparse

import torch

from niarb import atlas, nn
from niarb.nn.modules import frame
from niarb.tensors import periodic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", help="JSON/CSV affine map derived from Allen atlas data.")
    args = parser.parse_args()

    visual_map = (
        atlas.AllenAffineVisualFieldMap(args.map)
        if args.map
        else atlas.IdentityVisualFieldMap()
    )
    kernel = nn.VisualFieldTuning(nn.Scalar(0.5), visual_map, ["space", "ori"])

    x = frame.ParameterFrame(
        {
            "space": torch.tensor([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]]),
            "ori": periodic.tensor([[0.0], [45.0], [90.0]], extents=[(-90.0, 90.0)]),
        }
    )
    y = frame.ParameterFrame(
        {
            "space": torch.tensor([[100.0, 0.0], [100.0, 100.0], [0.0, 0.0]]),
            "ori": periodic.tensor([[0.0], [45.0], [90.0]], extents=[(-90.0, 90.0)]),
        }
    )

    with torch.no_grad():
        print(kernel(x, y))


if __name__ == "__main__":
    main()
