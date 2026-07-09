import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import torch

from niarb import neurons, nn
from mpl_config import set_rcParams


def sorted_fit_paths(fits_dir: Path) -> list[Path]:
    paths = list(fits_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No fitted parameter files found in {fits_dir}.")
    return sorted(paths, key=lambda path: float(path.stem))


def load_model_state(path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    out = {}
    for name in ("gW", "sigma", "kappa", "vf"):
        if name in state:
            out[name] = state[name]
            continue
        matches = [value for key, value in state.items() if key.endswith(f".{name}")]
        if matches:
            out[name] = matches[0]
    missing = {"gW", "sigma", "kappa"} - set(out)
    if missing:
        raise KeyError(f"Could not find {sorted(missing)} in fitted state dict {path}.")
    return out


def make_grid(N_space, N_ori, space_extent, cell_types):
    x = neurons.as_grid(
        len(cell_types),
        N_space=N_space,
        N_ori=N_ori,
        cell_types=cell_types,
        space_extent=(space_extent, space_extent),
    )
    x["dh"] = torch.zeros(x.shape)
    return x


def nearest_index(values: torch.Tensor, target: float) -> int:
    return int((values - target).abs().argmin().item())


def perturb_origin_horizontal(x, cell_type, dh):
    cell_types = list(x["cell_type"].categories)
    cell_idx = cell_types.index(cell_type)
    space = x["space"][0, :, :, 0]
    ori = x["ori"][0, 0, 0, :].tensor.squeeze(-1)
    ix = nearest_index(space[:, 0, 0], 0.0)
    iy = nearest_index(space[0, :, 1], 0.0)
    iori = nearest_index(ori, 0.0)
    x["dh"][cell_idx, ix, iy, iori] = dh
    return cell_idx, ix, iy, iori


def make_model(state, mode_name, seed=None):
    kwargs = {}
    if mode_name == "direct_space":
        kwargs = {"use_psi": True, "psi_mode": "direct_space"}
    model = nn.V1(
        ["cell_type", "space", "ori"],
        cell_types=["PYR", "PV"],
        tau=[1.0, 0.5],
        mode="matrix",
        seed=seed,
        **kwargs,
    )
    model.double()
    model.load_state_dict(state, strict=False)
    return model


def response(model, x):
    with torch.inference_mode():
        out = model(x.double(), output="response", ndim=x.ndim, to_dataframe=False)
    return out["dr"].detach().cpu()


def plot_responses(x, responses, cell_type, out, dpi):
    space_x = x["space"][0, :, 0, 0, 0].detach().cpu()
    space_y = x["space"][0, 0, :, 0, 1].detach().cpu()
    ori = x["ori"][0, 0, 0, :].tensor.squeeze(-1).detach().cpu()
    cell_types = list(x["cell_type"].categories)
    cell_idx = cell_types.index(cell_type)

    panels = [dr[cell_idx] for dr in responses.values()]
    vmax = max(float(panel.abs().max()) for panel in panels)
    if vmax == 0.0:
        vmax = 1.0
    norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    nrows, ncols = len(responses), len(ori)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(1.65 * ncols, 1.8 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )

    image = None
    extent = [
        float(space_x.min()),
        float(space_x.max()),
        float(space_y.min()),
        float(space_y.max()),
    ]
    for row, (model_name, dr) in enumerate(responses.items()):
        for col, theta in enumerate(ori):
            ax = axes[row, col]
            image = ax.imshow(
                dr[cell_idx, :, :, col].T,
                origin="lower",
                extent=extent,
                cmap="coolwarm",
                norm=norm,
                interpolation="nearest",
                aspect="equal",
            )
            if row == 0:
                ax.set_title(f"{float(theta):g} deg")
            if col == 0:
                ax.set_ylabel(f"{model_name}\ny (um)")
            if row == nrows - 1:
                ax.set_xlabel("x (um)")
            ax.axhline(0, color="black", linewidth=0.25, alpha=0.4)
            ax.axvline(0, color="black", linewidth=0.25, alpha=0.4)

    cbar = fig.colorbar(image, ax=axes, shrink=0.72)
    cbar.set_label(f"{cell_type} response")
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
    parser.add_argument("--N-ori", type=int, default=6)
    parser.add_argument("--space-extent", type=float, default=900.0)
    parser.add_argument("--cell-type", choices=["PYR", "PV"], default="PYR")
    parser.add_argument("--perturb-cell-type", choices=["PYR", "PV"], default="PYR")
    parser.add_argument("--dh", type=float, default=10000.0)
    parser.add_argument("--max-neurons", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--experiment-name", default="origin_horizontal_perturbation")
    parser.add_argument("--out", type=Path, default=Path("origin_horizontal_response.pdf"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    if any(N % 2 for N in args.N_space):
        parser.error("--N-space values must be even so the grid contains the origin.")
    if args.N_ori % 2:
        parser.error("--N-ori must be even so the grid contains horizontal ori=0.")
    if args.fit is not None and args.fit_index != 0:
        parser.error("Use either --fit or --fit-index, not both.")
    N_total = 2 * args.N_space[0] * args.N_space[1] * args.N_ori
    if args.max_neurons is not None and N_total > args.max_neurons:
        memory_gib = N_total**2 * 8 / 1024**3
        parser.error(
            f"This run has {N_total} neurons and needs at least {memory_gib:.1f} GiB "
            "for one float64 dense matrix. Reduce --N-space/--N-ori or raise "
            "--max-neurons deliberately on a large-memory machine."
        )

    fit = args.fit or sorted_fit_paths(args.fits_dir)[args.fit_index]
    state = load_model_state(fit)
    seeds = args.seeds if args.seeds is not None else [args.seed]

    set_rcParams()
    for seed in seeds:
        torch.manual_seed(seed)
        x = make_grid(args.N_space, args.N_ori, args.space_extent, ["PYR", "PV"])
        perturb_origin_horizontal(x, args.perturb_cell_type, args.dh)
        responses = {
            "paper": response(make_model(state, "paper", seed=seed), x),
            "direct space": response(make_model(state, "direct_space", seed=seed), x),
        }
        out = Path("results") / args.experiment_name / f"seed_{seed}" / args.out.name
        fig = plot_responses(x, responses, args.cell_type, out, args.dpi)
        plt.close(fig)
        print(f"Saved {out} using fit {fit}.")


if __name__ == "__main__":
    main()
