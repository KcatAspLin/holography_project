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
    d_space = space - space[ix, iy]
    distance = d_space.norm(dim=-1)
    psi = torch.atan2(d_space[..., 1], d_space[..., 0]) * 180.0 / torch.pi
    origin_mask = torch.zeros(distance.shape, dtype=torch.bool)
    origin_mask[ix, iy] = True
    return rel_ori, distance, psi, origin_mask


def response_panel(dr, cell_idx, perturb_idx):
    panel = dr[cell_idx].clone()
    perturb_cell_idx, ix, iy, iori = perturb_idx
    if cell_idx == perturb_cell_idx:
        panel[ix, iy, iori] = torch.nan
    return panel


def padded_limits(values, *, lower_bound=None):
    values = torch.as_tensor(values)
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return (0.0, 1.0)

    vmin = float(finite.min())
    vmax = float(finite.max())
    if vmin == vmax:
        pad = 1.0 if vmin == 0.0 else abs(vmin) * 0.05
    else:
        pad = (vmax - vmin) * 0.05
    lower = vmin - pad
    if lower_bound is not None:
        lower = lower_bound
    return (lower, vmax + pad)


def profile_xy(panel, values, origin_mask, *, values_are_orientation=False):
    selected = panel[~origin_mask, :]
    if values_are_orientation:
        x = values.expand_as(selected)
    else:
        x = values[~origin_mask].unsqueeze(-1).expand_as(selected)

    x = x.reshape(-1)
    y = selected.reshape(-1)
    finite = torch.isfinite(x) & torch.isfinite(y)
    return x[finite], y[finite]


def mean_profile_line(xs, ys, *, bins=None, bin_range=None):
    if bins is not None:
        if bin_range is None:
            lower, upper = xs.min(), xs.max()
        else:
            lower, upper = bin_range
        edges = torch.linspace(lower, upper, bins + 1, dtype=xs.dtype)
        centers, mean_y = [], []
        for idx in range(bins):
            if idx == bins - 1:
                mask = (xs >= edges[idx]) & (xs <= edges[idx + 1])
            else:
                mask = (xs >= edges[idx]) & (xs < edges[idx + 1])
            if not mask.any():
                continue
            centers.append((edges[idx] + edges[idx + 1]) / 2.0)
            mean_y.append(ys[mask].mean())
        return torch.stack(centers), torch.stack(mean_y)

    unique_x = torch.unique(xs).sort().values
    mean_y = []
    for value in unique_x:
        selected = ys[xs == value]
        mean_y.append(selected.mean())
    return unique_x, torch.stack(mean_y)


def plot_scatter_profile(
    x,
    responses,
    response_cell_type,
    perturb_idx,
    values,
    xlabel,
    title,
    out,
    dpi,
    *,
    values_are_orientation=False,
    x_lower_bound=None,
    mean_bins=None,
):
    _, _, _, origin_mask = grid_metadata(x, perturb_idx)
    cell_types = list(x["cell_type"].categories)
    cell_idx = cell_types.index(response_cell_type)

    points = []
    for model_name, dr in responses.items():
        panel = response_panel(dr, cell_idx, perturb_idx)
        points.append(
            (
                model_name,
                *profile_xy(
                    panel,
                    values,
                    origin_mask,
                    values_are_orientation=values_are_orientation,
                ),
            )
        )

    all_x = torch.cat([point[1] for point in points if point[1].numel()])
    all_y = torch.cat([point[2] for point in points if point[2].numel()])
    xlim = padded_limits(all_x, lower_bound=x_lower_bound)
    ylim = padded_limits(all_y)

    n_scatter_rows = len(points)
    fig, axes = plt.subplots(
        n_scatter_rows + 1,
        1,
        figsize=(5.8, 1.45 * (n_scatter_rows + 1)),
        sharex=False,
        sharey=False,
        constrained_layout=True,
        squeeze=False,
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for n, (model_name, xs, ys) in enumerate(points):
        ax = axes[n, 0]
        color = colors[n % len(colors)]
        ax.scatter(xs, ys, s=5, alpha=0.55, linewidths=0, color=color)
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.6)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_ylabel(f"{model_name}\n{response_cell_type}")

    mean_ax = axes[-1, 0]
    for n, (model_name, xs, ys) in enumerate(points):
        color = colors[n % len(colors)]
        line_x, mean_y = mean_profile_line(xs, ys, bins=mean_bins)
        mean_ax.plot(line_x, mean_y, linewidth=1.2, label=model_name, color=color)
    mean_ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.6)
    mean_ax.set_ylabel(f"mean\n{response_cell_type}")
    mean_ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    axes[0, 0].set_title(title)
    axes[-1, 0].set_xlabel(xlabel)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    return fig


def response_limits(values):
    lower, upper = padded_limits(values)
    lower = min(lower, 0.0)
    upper = max(upper, 0.0)
    return lower, upper


def response_to_radius(values, limits):
    return values - limits[0]


def format_radar_response_axis(ax, limits):
    lower, upper = limits
    radius_max = upper - lower
    ax.set_ylim(0.0, radius_max)
    ticks = torch.linspace(lower, upper, 5)
    ax.set_yticks((ticks - lower).tolist())
    ax.set_yticklabels([f"{float(tick):g}" for tick in ticks])
    ax.set_thetagrids(range(0, 360, 45))
    ax.set_rlabel_position(135)
    ax.tick_params(labelsize=6)

    zero_radius = -lower
    if 0.0 <= zero_radius <= radius_max:
        theta = torch.linspace(0.0, 2.0 * torch.pi, 361)
        radius = torch.full_like(theta, zero_radius)
        ax.plot(theta, radius, color="black", linewidth=0.6, alpha=0.6)


def close_radar_line(theta, radius):
    if theta.numel() == 0:
        return theta, radius
    return (
        torch.cat([theta, theta[:1] + 2.0 * torch.pi]),
        torch.cat([radius, radius[:1]]),
    )


def plot_psi_radar_profile(
    responses,
    response_cell_type,
    perturb_idx,
    psi,
    distance,
    space_extent,
    title,
    out,
    dpi,
    *,
    mean_bins=24,
):
    cell_idx = CELL_TYPES.index(response_cell_type)
    origin_mask = distance == 0.0
    spatial_mask = (~origin_mask) & (distance <= space_extent / 2.0)

    points = []
    for model_name, dr in responses.items():
        panel = response_panel(dr, cell_idx, perturb_idx)
        points.append((model_name, *profile_xy(panel, psi, ~spatial_mask)))

    all_y = torch.cat([point[2] for point in points if point[2].numel()])
    scatter_limits = response_limits(all_y)

    mean_lines = []
    for model_name, xs, ys in points:
        line_x, mean_y = mean_profile_line(
            xs, ys, bins=mean_bins, bin_range=(-180.0, 180.0)
        )
        mean_lines.append((model_name, line_x, mean_y))
    all_mean_y = torch.cat([line[2] for line in mean_lines if line[2].numel()])
    mean_limits = response_limits(all_mean_y)

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(8.0, 8.8),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    axes = axes.reshape(-1)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for n, (model_name, xs, ys) in enumerate(points):
        ax = axes[n]
        color = colors[n % len(colors)]
        ax.scatter(
            torch.deg2rad(xs),
            response_to_radius(ys, scatter_limits),
            s=5,
            alpha=0.55,
            linewidths=0,
            color=color,
        )
        format_radar_response_axis(ax, scatter_limits)
        ax.set_title(f"{model_name}\n{response_cell_type}", pad=10)

    mean_ax = axes[-1]
    for n, (model_name, line_x, mean_y) in enumerate(mean_lines):
        color = colors[n % len(colors)]
        theta, radius = close_radar_line(
            torch.deg2rad(line_x), response_to_radius(mean_y, mean_limits)
        )
        mean_ax.plot(theta, radius, linewidth=1.2, label=model_name, color=color)
    format_radar_response_axis(mean_ax, mean_limits)
    mean_ax.set_title(f"mean\n{response_cell_type}", pad=10)
    mean_ax.legend(loc="center left", bbox_to_anchor=(1.12, 0.5), frameon=False)

    fig.suptitle(title)
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
            rel_ori, distance, psi, _ = grid_metadata(x, perturb_idx)

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
                        "_over_distance.pdf"
                    )
                )
                fig = plot_scatter_profile(
                    x,
                    responses,
                    response_cell_type,
                    perturb_idx,
                    distance,
                    "Distance from perturbed neuron (um)",
                    (
                        f"perturb {perturb_cell_type}, {response_cell_type} response, "
                        "response over distance"
                    ),
                    out,
                    args.dpi,
                    x_lower_bound=0.0,
                )
                plt.close(fig)
                print(f"Saved {out} using fit {fit}.")

                out = (
                    seed_dir
                    / (
                        f"perturb_{perturb_cell_type}_response_{response_cell_type}"
                        "_over_orientation.pdf"
                    )
                )
                fig = plot_scatter_profile(
                    x,
                    responses,
                    response_cell_type,
                    perturb_idx,
                    rel_ori,
                    "Preferred orientation difference (deg)",
                    (
                        f"perturb {perturb_cell_type}, {response_cell_type} response, "
                        "response over preferred orientation"
                    ),
                    out,
                    args.dpi,
                    values_are_orientation=True,
                )
                plt.close(fig)
                print(f"Saved {out} using fit {fit}.")

                out = (
                    seed_dir
                    / (
                        f"perturb_{perturb_cell_type}_response_{response_cell_type}"
                        "_over_psi.pdf"
                    )
                )
                fig = plot_psi_radar_profile(
                    responses,
                    response_cell_type,
                    perturb_idx,
                    psi,
                    distance,
                    args.space_extent,
                    (
                        f"perturb {perturb_cell_type}, {response_cell_type} response, "
                        "response over psi"
                    ),
                    out,
                    args.dpi,
                    mean_bins=24,
                )
                plt.close(fig)
                print(f"Saved {out} using fit {fit}.")


if __name__ == "__main__":
    main()
