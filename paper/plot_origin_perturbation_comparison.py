import argparse
import gc
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


MODEL_VARIANTS = (
    (
        "original paper\ngamma=0",
        {"use_psi": True, "psi_mode": "direct_space", "psi_gamma": 0.0},
    ),
    (
        "direct mapping\ngamma=1",
        {"use_psi": True, "psi_mode": "direct_space", "psi_gamma": 1.0},
    ),
)


def make_model(state, model_kwargs, seed=None):
    kwargs = {}
    kwargs.update(model_kwargs)
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


def compute_responses(state, x, seed):
    responses = {}
    for label, model_kwargs in MODEL_VARIANTS:
        model = make_model(state, model_kwargs, seed=seed)
        responses[label] = response(model, x)
        del model
        gc.collect()
    return responses


def scale_responses(responses, scale):
    return {label: dr * scale for label, dr in responses.items()}


def masked_panels(responses, cell_idx, perturb_idx):
    panels = {}
    perturb_cell_idx, ix, iy, iori = perturb_idx
    for name, dr in responses.items():
        panel = dr[cell_idx].clone()
        if cell_idx == perturb_cell_idx:
            panel[ix, iy, iori] = torch.nan
        panels[name] = panel
    return panels


def centered_slice(center, width, size):
    if width is None or width >= size:
        return slice(0, size)

    start = center - width // 2
    end = start + width
    if start < 0:
        start, end = 0, width
    if end > size:
        start, end = size - width, size
    return slice(start, end)


def extent_slice(values, center, width):
    center_value = values[center]
    half_width = width / 2.0
    selected = torch.nonzero(
        (values >= center_value - half_width) & (values <= center_value + half_width)
    ).reshape(-1)
    if selected.numel() == 0:
        return slice(center, center + 1)
    return slice(int(selected[0]), int(selected[-1]) + 1)


def heatmap_slices(
    perturb_idx,
    N_space,
    heatmap_N_space=None,
    *,
    space_x=None,
    space_y=None,
    heatmap_extent=None,
):
    _, ix, iy, _ = perturb_idx
    if heatmap_extent is not None:
        if space_x is None or space_y is None:
            raise ValueError("space_x and space_y are required with heatmap_extent.")
        return (
            extent_slice(space_x, ix, heatmap_extent[0]),
            extent_slice(space_y, iy, heatmap_extent[1]),
        )
    if heatmap_N_space is None:
        return slice(0, N_space[0]), slice(0, N_space[1])
    return (
        centered_slice(ix, heatmap_N_space[0], N_space[0]),
        centered_slice(iy, heatmap_N_space[1], N_space[1]),
    )


def plot_responses(
    x,
    responses,
    cell_type,
    perturb_idx,
    out,
    dpi,
    heatmap_N_space=None,
    heatmap_extent=None,
):
    space_x = x["space"][0, :, 0, 0, 0].detach().cpu()
    space_y = x["space"][0, 0, :, 0, 1].detach().cpu()
    ori = x["ori"][0, 0, 0, :].tensor.squeeze(-1).detach().cpu()
    cell_types = list(x["cell_type"].categories)
    cell_idx = cell_types.index(cell_type)
    xs, ys = heatmap_slices(
        perturb_idx,
        (len(space_x), len(space_y)),
        heatmap_N_space,
        space_x=space_x,
        space_y=space_y,
        heatmap_extent=heatmap_extent,
    )
    plot_space_x = space_x[xs]
    plot_space_y = space_y[ys]

    panels = masked_panels(responses, cell_idx, perturb_idx)
    plot_panels = {name: panel[xs, ys, :] for name, panel in panels.items()}
    finite_parts = [
        panel[torch.isfinite(panel)].reshape(-1)
        for panel in plot_panels.values()
        if torch.isfinite(panel).any()
    ]
    finite_values = torch.cat(finite_parts) if finite_parts else torch.tensor([])
    vmin = float(finite_values.min()) if finite_values.numel() else 0.0
    vmax = float(finite_values.max()) if finite_values.numel() else 1.0
    if vmin == vmax:
        vmax = vmin + 1.0
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")

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
        float(plot_space_x.min()),
        float(plot_space_x.max()),
        float(plot_space_y.min()),
        float(plot_space_y.max()),
    ]
    for row, (model_name, panel) in enumerate(plot_panels.items()):
        for col, theta in enumerate(ori):
            ax = axes[row, col]
            image = ax.imshow(
                panel[:, :, col].T,
                origin="lower",
                extent=extent,
                cmap=cmap,
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


def model_slug(model_name):
    return "_".join(
        "".join(ch.lower() if ch.isalnum() else " " for ch in model_name).split()
    )


def plot_response_strength_heatmap(
    x,
    model_name,
    response_items,
    cell_type,
    out,
    dpi,
    heatmap_N_space=None,
    heatmap_extent=None,
):
    space_x = x["space"][0, :, 0, 0, 0].detach().cpu()
    space_y = x["space"][0, 0, :, 0, 1].detach().cpu()
    ori = x["ori"][0, 0, 0, :].tensor.squeeze(-1).detach().cpu()
    cell_types = list(x["cell_type"].categories)
    cell_idx = cell_types.index(cell_type)

    _, _, first_perturb_idx = response_items[0]
    xs, ys = heatmap_slices(
        first_perturb_idx,
        (len(space_x), len(space_y)),
        heatmap_N_space,
        space_x=space_x,
        space_y=space_y,
        heatmap_extent=heatmap_extent,
    )
    plot_space_x = space_x[xs]
    plot_space_y = space_y[ys]

    panels = []
    for dh, dr, perturb_idx in response_items:
        panel = response_panel_for_heatmap(dr, cell_idx, perturb_idx)
        panels.append((dh, panel[xs, ys, :]))

    all_orientation_panels = [
        torch.nanmean(panel, dim=-1) for _, panel in panels
    ]
    finite_parts = []
    for (_, panel), all_orientation_panel in zip(
        panels, all_orientation_panels, strict=True
    ):
        if torch.isfinite(panel).any():
            finite_parts.append(panel[torch.isfinite(panel)].reshape(-1))
        if torch.isfinite(all_orientation_panel).any():
            finite_parts.append(
                all_orientation_panel[
                    torch.isfinite(all_orientation_panel)
                ].reshape(-1)
            )
    finite_values = torch.cat(finite_parts) if finite_parts else torch.tensor([])
    vmin = float(finite_values.min()) if finite_values.numel() else 0.0
    vmax = float(finite_values.max()) if finite_values.numel() else 1.0
    if vmin == vmax:
        vmax = vmin + 1.0
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")

    nrows, ncols = len(panels), len(ori) + 1
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(1.65 * ncols, 1.7 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )

    extent = [
        float(plot_space_x.min()),
        float(plot_space_x.max()),
        float(plot_space_y.min()),
        float(plot_space_y.max()),
    ]
    image = None
    for row, ((dh, panel), all_orientation_panel) in enumerate(
        zip(panels, all_orientation_panels, strict=True)
    ):
        for col, theta in enumerate(ori):
            ax = axes[row, col]
            image = ax.imshow(
                panel[:, :, col].T,
                origin="lower",
                extent=extent,
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                aspect="equal",
            )
            if row == 0:
                ax.set_title(f"{float(theta):g} deg")
            if col == 0:
                ax.set_ylabel(f"dh={dh:g}\ny (um)")
            if row == nrows - 1:
                ax.set_xlabel("x (um)")
            ax.axhline(0, color="black", linewidth=0.25, alpha=0.4)
            ax.axvline(0, color="black", linewidth=0.25, alpha=0.4)
        ax = axes[row, -1]
        image = ax.imshow(
            all_orientation_panel.T,
            origin="lower",
            extent=extent,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
        if row == 0:
            ax.set_title("all ori\nmean")
        if row == nrows - 1:
            ax.set_xlabel("x (um)")
        ax.axhline(0, color="black", linewidth=0.25, alpha=0.4)
        ax.axvline(0, color="black", linewidth=0.25, alpha=0.4)

    fig.suptitle(model_name)
    cbar = fig.colorbar(image, ax=axes, shrink=0.72)
    cbar.set_label(f"{cell_type} perturbed - baseline activity")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    return fig


def response_panel_for_heatmap(dr, cell_idx, perturb_idx):
    panel = dr[cell_idx].clone()
    perturb_cell_idx, ix, iy, iori = perturb_idx
    if cell_idx == perturb_cell_idx:
        panel[ix, iy, iori] = torch.nan
    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fits-dir",
        type=Path,
        default=Path("paper/model_fits/no_disorder/fits"),
    )
    parser.add_argument("--fit", type=Path)
    parser.add_argument("--fit-index", type=int, default=0)
    parser.add_argument("--N-space", type=int, nargs=2, default=(48, 48))
    parser.add_argument("--heatmap-N-space", type=int, nargs=2)
    parser.add_argument(
        "--heatmap-extent",
        type=float,
        nargs=2,
        default=(100.0, 100.0),
        metavar=("WIDTH_UM", "HEIGHT_UM"),
        help="Physical heatmap window centered on the perturbation.",
    )
    parser.add_argument("--N-ori", type=int, default=6)
    parser.add_argument("--space-extent", type=float, default=400.0)
    parser.add_argument("--cell-type", choices=["PYR", "PV"], default="PYR")
    parser.add_argument("--perturb-cell-type", choices=["PYR", "PV"], default="PYR")
    parser.add_argument("--dh", type=float, default=10000.0)
    parser.add_argument(
        "--dh-values",
        type=float,
        nargs="+",
        default=(-10000.0, -5000.0, 0.0, 5000.0, 10000.0),
        help="Perturbation strengths used as heatmap rows.",
    )
    parser.add_argument("--max-neurons", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--experiment-name", default="origin_horizontal_perturbation")
    parser.add_argument("--out", type=Path, default=Path("origin_horizontal_response.pdf"))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    if any(N % 2 for N in args.N_space):
        parser.error("--N-space values must be even so the grid contains the origin.")
    if args.heatmap_N_space is not None and any(N <= 0 for N in args.heatmap_N_space):
        parser.error("--heatmap-N-space values must be positive.")
    if args.heatmap_extent is not None and any(v <= 0 for v in args.heatmap_extent):
        parser.error("--heatmap-extent values must be positive.")
    if not args.dh_values:
        parser.error("--dh-values must contain at least one perturbation strength.")
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
        perturb_idx = perturb_origin_horizontal(x, args.perturb_cell_type, 1.0)
        unit_responses = compute_responses(state, x, seed)

        heatmap_items = {model_name: [] for model_name, _ in MODEL_VARIANTS}
        for dh in args.dh_values:
            for model_name, dr in scale_responses(unit_responses, dh).items():
                heatmap_items[model_name].append((dh, dr, perturb_idx))

        seed_dir = Path("results") / args.experiment_name / f"seed_{seed}"
        for model_name, items in heatmap_items.items():
            out = seed_dir / f"{args.out.stem}_{model_slug(model_name)}{args.out.suffix}"
            fig = plot_response_strength_heatmap(
                x,
                model_name,
                items,
                args.cell_type,
                out,
                args.dpi,
                args.heatmap_N_space,
                args.heatmap_extent,
            )
            plt.close(fig)
            print(f"Saved {out} using fit {fit}.")


if __name__ == "__main__":
    main()
