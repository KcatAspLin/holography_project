import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from mpl_config import set_rcParams
from plot_origin_perturbation_comparison import (
    MODEL_VARIANTS,
    load_model_state,
    make_grid,
    make_model,
    perturb_origin_horizontal,
    plot_responses,
    response,
    sorted_fit_paths,
)


CELL_TYPES = ("PYR", "PV")


def angle_diff_deg(x, y):
    return (x - y + 180.0) % 360.0 - 180.0


def validate_args(parser, args):
    if any(N % 2 for N in args.N_space):
        parser.error("--N-space values must be even so the grid contains the origin.")
    if args.N_ori % 2:
        parser.error("--N-ori must be even so the grid contains horizontal ori=0.")
    if args.fit is not None and args.fit_index != 0:
        parser.error("Use either --fit or --fit-index, not both.")
    if args.distance_tol is not None and args.distance_tol < 0:
        parser.error("--distance-tol must be non-negative.")
    if args.psi_tol < 0:
        parser.error("--psi-tol must be non-negative.")

    N_total = len(CELL_TYPES) * args.N_space[0] * args.N_space[1] * args.N_ori
    if args.max_neurons is not None and N_total > args.max_neurons:
        memory_gib = N_total**2 * 8 / 1024**3
        parser.error(
            f"This run has {N_total} neurons and needs at least {memory_gib:.1f} GiB "
            "for one float64 dense matrix. Reduce --N-space/--N-ori or raise "
            "--max-neurons deliberately on a large-memory machine."
        )


def compute_responses(state, x, seed):
    return {
        label: response(make_model(state, model_kwargs, seed=seed), x)
        for label, model_kwargs in MODEL_VARIANTS
    }


def grid_metadata(x, perturb_idx):
    _, ix, iy, iori = perturb_idx
    space = x["space"][0, :, :, 0].detach().cpu()
    ori = x["ori"][0, 0, 0, :].tensor.squeeze(-1).detach().cpu()
    perturb_ori = float(ori[iori])
    rel_ori = torch.as_tensor(
        angle_diff_deg(ori.numpy(), perturb_ori), dtype=ori.dtype
    )
    distance = space.norm(dim=-1)
    psi = torch.atan2(space[..., 1], space[..., 0]) * 180.0 / torch.pi
    origin_mask = torch.zeros(distance.shape, dtype=torch.bool)
    origin_mask[ix, iy] = True
    return rel_ori, distance, psi, origin_mask


def distance_mask(distance, origin_mask, target, tol):
    if tol is None:
        allowed = distance[~origin_mask]
        selected_distance = float(allowed[(allowed - target).abs().argmin()])
        mask = (distance - selected_distance).abs() <= 1.0e-5
    else:
        selected_distance = target
        mask = (distance - target).abs() <= tol
    mask = mask & ~origin_mask
    if not mask.any():
        raise ValueError(
            f"No neurons found near distance {target} with tolerance {tol}."
        )
    return mask, selected_distance


def psi_mask(psi, origin_mask, target, tol):
    diff = torch.as_tensor(angle_diff_deg(psi.numpy(), target), dtype=psi.dtype)
    mask = (diff.abs() <= tol) & ~origin_mask
    if not mask.any():
        raise ValueError(
            f"No neurons found near psi={target} degrees with tolerance {tol}."
        )
    return mask


def plot_orientation_profile(
    x,
    responses,
    response_cell_type,
    perturb_idx,
    spatial_mask,
    title,
    out,
    dpi,
):
    rel_ori, _, _, _ = grid_metadata(x, perturb_idx)
    cell_types = list(x["cell_type"].categories)
    cell_idx = cell_types.index(response_cell_type)
    perturb_cell_idx, ix, iy, iori = perturb_idx

    fig, ax = plt.subplots(figsize=(5.8, 3.6), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for n, (model_name, dr) in enumerate(responses.items()):
        panel = dr[cell_idx].clone()
        if cell_idx == perturb_cell_idx:
            panel[ix, iy, iori] = torch.nan
        selected = panel[spatial_mask, :]
        y = torch.nanmean(selected, dim=0)
        y_min = torch.nan_to_num(selected, nan=float("inf")).min(dim=0).values
        y_max = torch.nan_to_num(selected, nan=float("-inf")).max(dim=0).values
        color = colors[n % len(colors)]
        ax.plot(rel_ori, y, marker="o", linewidth=1.4, label=model_name, color=color)
        if selected.shape[0] > 1:
            ax.fill_between(rel_ori, y_min, y_max, color=color, alpha=0.12, linewidth=0)

    ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("Preferred orientation difference (deg)")
    ax.set_ylabel(f"{response_cell_type} response")
    ax.set_title(title)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
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
    parser.add_argument("--distance", type=float, default=50.0)
    parser.add_argument("--distance-tol", type=float)
    parser.add_argument("--separation-psi", type=float, default=0.0)
    parser.add_argument("--psi-tol", type=float, default=5.0)
    parser.add_argument("--dh", type=float, default=10000.0)
    parser.add_argument("--max-neurons", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--experiment-name",
        default="origin_horizontal_perturbation_celltypes",
    )
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
            _, distance, psi, origin_mask = grid_metadata(x, perturb_idx)
            d_mask, selected_distance = distance_mask(
                distance, origin_mask, args.distance, args.distance_tol
            )
            p_mask = psi_mask(
                psi, origin_mask, args.separation_psi, args.psi_tol
            )

            for response_cell_type in CELL_TYPES:
                out = (
                    seed_dir
                    / f"perturb_{perturb_cell_type}_response_{response_cell_type}.pdf"
                )
                fig = plot_responses(
                    x,
                    responses,
                    response_cell_type,
                    perturb_idx,
                    out,
                    args.dpi,
                )
                plt.close(fig)
                print(f"Saved {out} using fit {fit}.")

                out = (
                    seed_dir
                    / (
                        f"perturb_{perturb_cell_type}_response_{response_cell_type}"
                        "_ori_at_distance.pdf"
                    )
                )
                fig = plot_orientation_profile(
                    x,
                    responses,
                    response_cell_type,
                    perturb_idx,
                    d_mask,
                    (
                        f"perturb {perturb_cell_type}, {response_cell_type} response, "
                        f"d = {selected_distance:.3g} um"
                    ),
                    out,
                    args.dpi,
                )
                plt.close(fig)
                print(f"Saved {out} using fit {fit}.")

                out = (
                    seed_dir
                    / (
                        f"perturb_{perturb_cell_type}_response_{response_cell_type}"
                        "_ori_at_psi.pdf"
                    )
                )
                fig = plot_orientation_profile(
                    x,
                    responses,
                    response_cell_type,
                    perturb_idx,
                    p_mask,
                    (
                        f"perturb {perturb_cell_type}, {response_cell_type} response, "
                        f"psi = {args.separation_psi:g} deg"
                    ),
                    out,
                    args.dpi,
                )
                plt.close(fig)
                print(f"Saved {out} using fit {fit}.")


if __name__ == "__main__":
    main()
