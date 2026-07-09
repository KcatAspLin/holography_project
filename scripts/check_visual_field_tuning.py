#!/usr/bin/env python
"""Inspect the modified Eq. 8 visual-field orientation factor."""

import argparse

import torch

from niarb import atlas, nn
from niarb.nn.modules import frame
from niarb.tensors import periodic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-psi", action="store_true")
    parser.add_argument(
        "--psi-mode",
        choices=["independent", "visual_field", "direct_space"],
        default="independent",
    )
    parser.add_argument("--use-visual-field-tuning", action="store_true")
    parser.add_argument("--psi", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--map", help="JSON/CSV affine map derived from Allen atlas data.")
    args = parser.parse_args()

    if args.use_visual_field_tuning:
        if args.psi_mode != "independent":
            parser.error("--use-visual-field-tuning cannot be combined with --psi-mode")
        args.psi_mode = "visual_field"
    use_psi = args.use_psi or args.psi_mode != "independent"
    if args.psi_mode != "independent" and args.psi is not None:
        parser.error("--psi can only be used with --psi-mode independent")
    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.psi_mode == "visual_field":
        visual_map = (
            atlas.AllenAffineVisualFieldMap(args.map)
            if args.map
            else atlas.IdentityVisualFieldMap()
        )
        kernel = nn.VisualFieldTuning(nn.Scalar(0.5), visual_map, ["space", "ori"])
    elif args.psi_mode == "direct_space":
        kernel = nn.DirectSpaceTuning(nn.Scalar(0.5), ["space", "ori"])
    elif use_psi:
        kernel = nn.PsiTuning(nn.Scalar(0.5), args.psi, "ori")
    else:
        kernel = nn.Tuning(nn.Scalar(0.5), "ori")

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
