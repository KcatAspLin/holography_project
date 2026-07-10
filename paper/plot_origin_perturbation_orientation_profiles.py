import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from mpl_config import set_rcParams
from plot_origin_perturbation_celltype_comparison import (
    CELL_TYPES,
    compute_responses,
    validate_args,
)
from plot_origin_perturbation_comparison import (
    load_model_state,
    make_grid,
    perturb_origin_horizontal,
    sorted_fit_paths,
)


def space_grid(x):
    return x["space"][0, :, :, 0].detach().cpu()


def orientation_differences(x):
    return x["ori"][0, 0, 0, :].tensor.squeeze(-1).detach().cpu()


def format_float(value):
    return f"{value:g}".replace("-", "m").replace(".", "p")


def nearest_shell_mask(radius, distance):
    nearest = radius.flatten()[(radius - distance).abs().argmin()]
    return torch.isclose(radius, nearest), float(nearest)


def distance_mask(x, distance, width):
    space = space_grid(x)
    radius = torch.linalg.norm(space, dim=-1)
    if width is None:
        return nearest_shell_mask(radius, distance)
    return (radius - distance).abs() <= width / 2, distance


def angle_difference(angle, target):
    return torch.atan2(torch.sin(angle - target), torch.cos(angle - target))


def psi_mask(x, psi_degrees, width_degrees):
    space = space_grid(x)
    radius = torch.linalg.norm(space, dim=-1)
    angle = torch.atan2(space[..., 1], space[..., 0])
    target = math.radians(psi_degrees)
    abs_diff = angle_difference(angle, target).abs()
    non_origin = radius > 0

    if width_degrees is None:
        nearest = abs_diff[non_origin].min()
        return non_origin & torch.isclose(abs_diff, nearest), math.degrees(float(target))

    width = math.radians(width_degrees)
    return non_origin & (abs_diff <= width / 2), psi_degrees


def profile_from_mask(dr, cell_idx, mask, perturb_idx):
    perturb_cell_idx, ix, iy, iori = perturb_idx
    panel = dr[cell_idx].clone()
    if cell_idx == perturb_cell_idx:
        panel[ix, iy, iori] = torch.nan
    selected = panel[mask]
    return torch.nanmean(selected, dim=0)


def plot_orientation_profiles(x, responses, response_cell_type, perturb_idx, mask, title, out, dpi):
    cell_types = list(x["cell_type"].categories)
    cell_idx = cell_types.index(response_cell_type)
    dori = orientation_differences(x)

    if not mask.any():
        raise ValueError(f"No spatial grid points selected for {title}.")

    fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
    for model_name, dr in responses.items():
        y = profile_from_mask(dr, cell_idx, mask, perturb_idx)
        ax.plot(dori, y, marker="o", linewidth=1.2, markersize=3, label=model_name)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Preferred orientation difference from perturbation (deg)")
    ax.set_ylabel(f"{response_cell_type} response")
    ax.set_title(title)
    ax.legend(fontsize=6, frameon=False)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fits-dir",
        type=Path,
        default=Path("paper/model_fits/no_disorder/fits"),
    )
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--fit-index", type=int, default=0)
    parser.add_argument("--N-space", type=int, nargs=2, default=(16, 16))
    parser.add_argument("--N-ori", type=int, default=8)
    parser.add_argument("--space-extent", type=float, default=200.0)
    parser.add_argument("--dh", type=float, default=10000.0)
    parser.add_argument("--max-neurons", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--experiment-name",
        default="origin_horizontal_perturbation_orientation_profiles",
    )
    parser.add_argument("--distance", type=float, default=50.0)
    parser.add_argument("--distance-width", type=float)
    parser.add_argument("--psi", type=float, default=0.0)
    parser.add_argument("--psi-width", type=float)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    validate_args(parser, args)

    fit = args.fit or sorted_fit_paths(args.fits_dir)[args.fit_index]
    state = load_model_state(fit)
    seeds = args.seeds if args.seeds is not None else [args.seed]

    set_rcParams()
    for seed in seeds:
        torch.manual_seed(seed)
        seed_dir = Path("results") / args.experiment_name / f"seed_{seed}"

        for perturb_cell_type in CELL_TYPES:
            x = make_grid(args.N_space, args.N_ori, args.space_extent, CELL_TYPES)
            perturb_idx = perturb_origin_horizontal(x, perturb_cell_type, args.dh)
            responses = compute_responses(state, x, seed)

            dist_mask, actual_distance = distance_mask(
                x, args.distance, args.distance_width
            )
            angle_mask, actual_psi = psi_mask(x, args.psi, args.psi_width)

            for response_cell_type in CELL_TYPES:
                dist_out = (
                    seed_dir
                    / (
                        f"distance_d{format_float(actual_distance)}_"
                        f"perturb_{perturb_cell_type}_response_{response_cell_type}.pdf"
                    )
                )
                fig = plot_orientation_profiles(
                    x,
                    responses,
                    response_cell_type,
                    perturb_idx,
                    dist_mask,
                    f"{perturb_cell_type} perturbation, distance {actual_distance:g} um",
                    dist_out,
                    args.dpi,
                )
                plt.close(fig)
                print(f"Saved {dist_out} using fit {fit}.")

                psi_out = (
                    seed_dir
                    / (
                        f"psi_{format_float(actual_psi)}deg_"
                        f"perturb_{perturb_cell_type}_response_{response_cell_type}.pdf"
                    )
                )
                fig = plot_orientation_profiles(
                    x,
                    responses,
                    response_cell_type,
                    perturb_idx,
                    angle_mask,
                    f"{perturb_cell_type} perturbation, psi {actual_psi:g} deg",
                    psi_out,
                    args.dpi,
                )
                plt.close(fig)
                print(f"Saved {psi_out} using fit {fit}.")


if __name__ == "__main__":
    main()
