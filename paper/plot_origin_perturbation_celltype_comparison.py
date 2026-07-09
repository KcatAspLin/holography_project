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


def validate_args(parser, args):
    if any(N % 2 for N in args.N_space):
        parser.error("--N-space values must be even so the grid contains the origin.")
    if args.N_ori % 2:
        parser.error("--N-ori must be even so the grid contains horizontal ori=0.")
    if args.fit is not None and args.fit_index != 0:
        parser.error("Use either --fit or --fit-index, not both.")

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


if __name__ == "__main__":
    main()
