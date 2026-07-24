import argparse
import gc
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import torch

from mpl_config import set_rcParams
from niarb.nn.modules import frame
from niarb.tensors import categorical, periodic
from plot_origin_perturbation_comparison import (
    MODEL_VARIANTS,
    load_model_state,
    make_model,
    model_slug,
    nearest_index,
    perturbation_label,
    sorted_fit_paths,
)


CELL_TYPES = ("PYR", "PV")


def angle_diff_deg(x, y):
    return (x - y + 180.0) % 360.0 - 180.0


def validate_args(parser, args):
    if args.N_space[0] < 2:
        parser.error("--N-space radial count must be at least 2.")
    if args.N_space[1] < 3:
        parser.error("--N-space angular count must be at least 3.")
    if args.N_ori % 2:
        parser.error("--N-ori must be even so the grid contains horizontal ori=0.")
    if not args.dh_values:
        parser.error("--dh-values must contain at least one perturbation strength.")
    if args.fit is not None and args.fit_index != 0:
        parser.error("Use either --fit or --fit-index, not both.")

    N_spatial = 1 + (args.N_space[0] - 1) * args.N_space[1]
    N_total = len(CELL_TYPES) * N_spatial * args.N_ori
    if args.max_neurons is not None and N_total > args.max_neurons:
        memory_gib = N_total**2 * 8 / 1024**3
        parser.error(
            f"This run has {N_total} neurons and needs at least {memory_gib:.1f} GiB "
            "for one float64 dense matrix. Reduce --N-space/--N-ori or raise "
            "--max-neurons deliberately on a large-memory machine."
        )


def polar_points(N_radial, N_angle, space_extent):
    radius_max = space_extent / 2.0
    radii = torch.linspace(0.0, radius_max, N_radial)
    angles = torch.linspace(-180.0, 180.0, N_angle + 1)[:-1]
    angles_rad = torch.deg2rad(angles)

    points = [torch.zeros(2)]
    point_radii = [torch.tensor(0.0)]
    point_angles = [torch.tensor(0.0)]
    space_dV = [math.pi * (radii[1].item() / 2.0) ** 2]

    for idx, radius in enumerate(radii[1:], start=1):
        if idx == 1:
            inner = radii[1] / 2.0
        else:
            inner = (radii[idx - 1] + radius) / 2.0

        if idx == N_radial - 1:
            outer = radii[-1]
        else:
            outer = (radius + radii[idx + 1]) / 2.0

        sector_area = math.pi * float(outer**2 - inner**2) / N_angle
        for angle, angle_rad in zip(angles, angles_rad):
            points.append(
                radius * torch.stack([torch.cos(angle_rad), torch.sin(angle_rad)])
            )
            point_radii.append(radius)
            point_angles.append(angle)
            space_dV.append(sector_area)

    return {
        "space": torch.stack(points),
        "radius": torch.stack(point_radii),
        "angle": torch.stack(point_angles),
        "space_dV": torch.tensor(space_dV),
    }


def make_polar_grid(N_space, N_ori, space_extent, cell_types):
    N_radial, N_angle = N_space
    polar = polar_points(N_radial, N_angle, space_extent)
    n_cell_types = len(cell_types)
    n_points = polar["space"].shape[0]

    cell_type = categorical.as_tensor(
        torch.arange(n_cell_types), categories=cell_types
    ).reshape(n_cell_types, 1, 1)
    ori = periodic.linspace(-90.0, 90.0, N_ori).reshape(1, 1, N_ori, 1)
    space = polar["space"].reshape(1, n_points, 1, 2)
    space_dV = polar["space_dV"].reshape(1, n_points, 1)
    dV = space_dV * float(ori.period.prod()) / N_ori

    x = frame.ParameterFrame(
        {
            "cell_type": cell_type,
            "space": space,
            "ori": ori,
            "dV": dV,
            "space_dV": space_dV,
        },
        ndim=3,
    )
    x["dh"] = torch.zeros(x.shape)
    return x, polar


def perturb_origin_horizontal(x, cell_type, dh):
    cell_types = list(x["cell_type"].categories)
    cell_idx = cell_types.index(cell_type)
    ori = x["ori"][0, 0, :].tensor.squeeze(-1)
    iori = nearest_index(ori, 0.0)
    point_idx = 0
    x["dh"][cell_idx, point_idx, iori] = dh
    return cell_idx, point_idx, iori


def baseline_activity(model, x):
    with torch.inference_mode():
        baseline = model.f(model.vf).detach().cpu()

    if baseline.ndim == 0:
        return torch.full(x.shape, float(baseline), dtype=baseline.dtype)

    if baseline.numel() == len(x["cell_type"].categories):
        cell_idx = x["cell_type"].detach().cpu().to(torch.long)
        return baseline.reshape(-1)[cell_idx]

    return baseline.broadcast_to(x.shape).detach().cpu()


def activity_difference(model, x):
    with torch.inference_mode():
        out = model(x.double(), output="response", ndim=x.ndim, to_dataframe=False)
    response_change = out["dr"].detach().cpu()
    baseline = baseline_activity(model, x)
    perturbed = baseline + response_change
    return perturbed - baseline


def compute_responses(state, x, seed, mode="matrix", approx_order=2):
    responses = {}
    for label, model_kwargs in MODEL_VARIANTS:
        model = make_model(
            state, model_kwargs, seed=seed, mode=mode, approx_order=approx_order
        )
        responses[label] = activity_difference(model, x)
        del model
        gc.collect()
    return responses


def scale_responses(responses, scale):
    return {label: dr * scale for label, dr in responses.items()}


def perturbation_sign(dh):
    if dh > 0:
        return "positive"
    if dh < 0:
        return "negative"
    return "zero"


def default_dh_values(dh):
    return tuple(scale * dh for scale in (-2.0, -1.0, 0.0, 1.0, 2.0))


def write_perturbation_note(seed_dir, args, fit, seed):
    seed_dir.mkdir(parents=True, exist_ok=True)
    sign = perturbation_sign(args.dh)
    note = (
        f"fit={fit}\n"
        f"seed={seed}\n"
        f"dh={args.dh:g}\n"
        f"dh_sign={sign}\n"
        f"dh_values={' '.join(f'{dh:g}' for dh in args.dh_values)}\n"
        f"dh_value_signs={' '.join(perturbation_sign(dh) for dh in args.dh_values)}\n"
        f"polar_circle_radius_um={args.space_extent / 2:g}\n"
        f"response_mode={args.response_mode}\n"
        f"approx_order={args.approx_order}\n"
        "distance_profile_layout=one file per model; rows=dh_values; over_distance=overall mean; over_distance_by_orientation=mean lines by orientation preference\n"
        "psi_profile_layout=one file per dh value; over_psi=model comparison; over_psi_by_orientation=columns=models, rows=distance, lines=orientation preference\n"
        "baseline_activity=model.f(model.vf)\n"
        "perturbed_activity=baseline_activity+model_response\n"
        "plotted_value=perturbed_activity-baseline_activity\n"
        "matrix_mode_note=matrix solves (I-W)dr=dh exactly; matrix_approx uses a finite Neumann series.\n"
    )
    (seed_dir / "perturbation_metadata.txt").write_text(note)
    print(
        f"Perturbation dh={args.dh:g} ({sign}); "
        f"wrote {seed_dir / 'perturbation_metadata.txt'}."
    )


def grid_metadata(x, perturb_idx):
    _, point_idx, iori = perturb_idx
    space = x["space"][0, :, 0].detach().cpu()
    ori = x["ori"][0, 0, :].tensor.squeeze(-1).detach().cpu()
    perturb_ori = float(ori[iori])
    rel_ori = torch.as_tensor(
        angle_diff_deg(ori.numpy(), perturb_ori), dtype=ori.dtype
    )
    d_space = space - space[point_idx]
    distance = d_space.norm(dim=-1)
    psi = torch.atan2(d_space[..., 1], d_space[..., 0]) * 180.0 / torch.pi
    origin_mask = distance <= 1.0e-8
    return rel_ori, distance, psi, origin_mask


def response_panel(dr, cell_idx, perturb_idx):
    panel = dr[cell_idx].clone()
    perturb_cell_idx, point_idx, iori = perturb_idx
    if cell_idx == perturb_cell_idx:
        panel[point_idx, iori] = torch.nan
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


def distance_orientation_mean_lines(panel, distance, origin_mask, ori, *, bins=32):
    distances = distance[~origin_mask].reshape(-1)
    bin_range = (0.0, float(distances.max())) if distances.numel() else None
    lines = []
    for ori_idx, theta in enumerate(ori):
        responses = panel[..., ori_idx][~origin_mask].reshape(-1)
        finite = torch.isfinite(distances) & torch.isfinite(responses)
        if not finite.any():
            continue
        line_x, mean_y = mean_profile_line(
            distances[finite], responses[finite], bins=bins, bin_range=bin_range
        )
        lines.append((float(theta), line_x, mean_y))
    return lines


def distance_overall_mean_line(panel, distance, origin_mask, *, bins=32):
    selected = panel[~origin_mask, :]
    distances = distance[~origin_mask].unsqueeze(-1).expand_as(selected)
    xs = distances.reshape(-1)
    ys = selected.reshape(-1)
    finite = torch.isfinite(xs) & torch.isfinite(ys)
    if not finite.any():
        return torch.tensor([]), torch.tensor([])
    return mean_profile_line(
        xs[finite],
        ys[finite],
        bins=bins,
        bin_range=(0.0, float(distance[~origin_mask].max())),
    )


def value_mask(values, target):
    tol = max(1.0e-4, abs(float(target)) * 1.0e-4)
    return (values - target).abs() <= tol


def selected_distance_values(distance, origin_mask, count=4):
    distances = torch.unique(distance[~origin_mask]).sort().values
    if distances.numel() <= count:
        return distances

    targets = torch.linspace(float(distances[0]), float(distances[-1]), count)
    selected = []
    for target in targets:
        idx = int((distances - target).abs().argmin())
        value = distances[idx]
        if not selected or not bool(
            value_mask(torch.as_tensor(selected[-1]), value)
        ):
            selected.append(value)
    return torch.stack(selected)


def orientation_values(x):
    return x["ori"][0, 0, :].tensor.squeeze(-1).detach().cpu()


def nearest_orientation_index(ori, target):
    return int(torch.as_tensor(angle_diff_deg(ori.numpy(), target)).abs().argmin())


def psi_summary_at_distance_orientation(panel, distance, psi, ori_idx, d):
    spatial_mask = value_mask(distance, d)
    psi_values = torch.unique(psi[spatial_mask]).sort().values
    x, y_mean, y_min, y_max = [], [], [], []
    for psi_value in psi_values:
        mask = spatial_mask & value_mask(psi, psi_value)
        selected = panel[mask, ori_idx].reshape(-1)
        selected = selected[torch.isfinite(selected)]
        if selected.numel() == 0:
            continue
        x.append(psi_value)
        y_mean.append(selected.mean())
        y_min.append(selected.min())
        y_max.append(selected.max())
    return (
        torch.stack(x),
        torch.stack(y_mean),
        torch.stack(y_min),
        torch.stack(y_max),
    )


def plot_psi_distance_profiles(
    x, responses, response_cell_type, perturb_idx, distance, psi, title, out, dpi
):
    _, _, _, origin_mask = grid_metadata(x, perturb_idx)
    cell_types = list(x["cell_type"].categories)
    cell_idx = cell_types.index(response_cell_type)
    distances = selected_distance_values(distance, origin_mask, count=4)
    ori = orientation_values(x)
    orientation_targets = (0.0, -90.0)
    orientation_indices = [
        nearest_orientation_index(ori, target) for target in orientation_targets
    ]

    summaries = []
    for d in distances:
        per_orientation = []
        for target, ori_idx in zip(orientation_targets, orientation_indices):
            per_model = []
            for model_name, dr in responses.items():
                panel = response_panel(dr, cell_idx, perturb_idx)
                per_model.append(
                    (
                        model_name,
                        *psi_summary_at_distance_orientation(
                            panel, distance, psi, ori_idx, d
                        ),
                    )
                )
            per_orientation.append((target, float(ori[ori_idx]), per_model))
        summaries.append((d, per_orientation))

    all_x = torch.cat(
        [
            summary[1]
            for _, per_orientation in summaries
            for _, _, per_model in per_orientation
            for summary in per_model
            if summary[1].numel()
        ]
    )
    all_y = torch.cat(
        [
            torch.cat([summary[3], summary[4]])
            for _, per_orientation in summaries
            for _, _, per_model in per_orientation
            for summary in per_model
            if summary[3].numel()
        ]
    )
    xlim = padded_limits(all_x)
    ylim = padded_limits(all_y)

    fig, axes = plt.subplots(
        len(summaries),
        len(orientation_targets),
        figsize=(8.6, 1.75 * len(summaries)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for row, (d, per_orientation) in enumerate(summaries):
        for col, (target, actual_ori, per_model) in enumerate(per_orientation):
            ax = axes[row, col]
            for n, (model_name, xs, mean_y, min_y, max_y) in enumerate(per_model):
                color = colors[n % len(colors)]
                ax.plot(
                    xs,
                    mean_y,
                    marker="o",
                    markersize=2.2,
                    linewidth=1.2,
                    label=model_name,
                    color=color,
                )
                ax.fill_between(
                    xs, min_y, max_y, color=color, alpha=0.12, linewidth=0
                )
            ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.6)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            if row == 0:
                ax.set_title(f"pref ori {actual_ori:g} deg")
            if col == 0:
                ax.set_ylabel(f"d={float(d):g} um\n{response_cell_type}")

    fig.suptitle(title)
    for ax in axes[-1, :]:
        ax.set_xlabel("Angle from perturbed neuron, psi (deg)")
    axes[0, -1].legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    return fig


def plot_responses_polar(x, responses, cell_type, perturb_idx, out, dpi):
    space = x["space"][0, :, 0].detach().cpu()
    ori = x["ori"][0, 0, :].tensor.squeeze(-1).detach().cpu()
    cell_types = list(x["cell_type"].categories)
    cell_idx = cell_types.index(cell_type)

    panels = {
        model_name: response_panel(dr, cell_idx, perturb_idx)
        for model_name, dr in responses.items()
    }
    finite_parts = [
        panel[torch.isfinite(panel)].reshape(-1)
        for panel in panels.values()
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

    radius_max = float(space.norm(dim=-1).max())
    image = None
    for row, (model_name, panel) in enumerate(panels.items()):
        for col, theta in enumerate(ori):
            ax = axes[row, col]
            image = ax.scatter(
                space[:, 0],
                space[:, 1],
                c=panel[:, col],
                s=16,
                cmap=cmap,
                norm=norm,
                linewidths=0,
            )
            if row == 0:
                ax.set_title(f"{float(theta):g} deg")
            if col == 0:
                ax.set_ylabel(f"{model_name}\ny (um)")
            if row == nrows - 1:
                ax.set_xlabel("x (um)")
            ax.set_xlim(-radius_max, radius_max)
            ax.set_ylim(-radius_max, radius_max)
            ax.set_aspect("equal")
            ax.axhline(0, color="black", linewidth=0.25, alpha=0.4)
            ax.axvline(0, color="black", linewidth=0.25, alpha=0.4)

    cbar = fig.colorbar(image, ax=axes, shrink=0.72)
    cbar.set_label(f"{cell_type} response")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    return fig


def plot_distance_orientation_profiles(
    x, responses, response_cell_type, perturb_idx, distance, title, out, dpi
):
    _, _, _, origin_mask = grid_metadata(x, perturb_idx)
    cell_types = list(x["cell_type"].categories)
    cell_idx = cell_types.index(response_cell_type)
    ori = orientation_values(x)

    profiles = []
    for model_name, dr in responses.items():
        panel = response_panel(dr, cell_idx, perturb_idx)
        profiles.append(
            (
                model_name,
                distance_orientation_mean_lines(panel, distance, origin_mask, ori),
            )
        )

    all_x = torch.cat(
        [
            line_x
            for _, lines in profiles
            for _, line_x, mean_y in lines
            if line_x.numel() and mean_y.numel()
        ]
    )
    all_y = torch.cat(
        [
            mean_y
            for _, lines in profiles
            for _, line_x, mean_y in lines
            if line_x.numel() and mean_y.numel()
        ]
    )
    xlim = padded_limits(all_x, lower_bound=0.0)
    ylim = padded_limits(all_y)

    nrows = len(profiles)
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(6.4, 1.7 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    cmap = plt.get_cmap("twilight_shifted")
    norm = colors.Normalize(vmin=float(ori.min()), vmax=float(ori.max()))
    for row, (model_name, lines) in enumerate(profiles):
        ax = axes[row, 0]
        for theta, line_x, mean_y in lines:
            ax.plot(
                line_x,
                mean_y,
                linewidth=1.2,
                color=cmap(norm(theta)),
            )
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.6)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_ylabel(f"{model_name}\n{response_cell_type}")

    axes[0, 0].set_title(f"{title}\nresponse = perturbed - baseline activity")
    axes[-1, 0].set_xlabel("Distance from perturbed neuron (um)")
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        shrink=0.8,
    )
    cbar.set_label("Preferred orientation (deg)")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    return fig


def plot_response_strength_distance_profiles(
    x,
    model_name,
    response_items,
    response_cell_type,
    distance,
    title,
    out,
    dpi,
    default_dh=10000.0,
):
    _, _, _, origin_mask = grid_metadata(x, response_items[0][2])
    cell_types = list(x["cell_type"].categories)
    cell_idx = cell_types.index(response_cell_type)

    profiles = []
    for dh, dr, perturb_idx in response_items:
        panel = response_panel(dr, cell_idx, perturb_idx)
        profiles.append((dh, *distance_overall_mean_line(panel, distance, origin_mask)))

    all_x = torch.cat([line_x for _, line_x, mean_y in profiles if line_x.numel()])
    all_y = torch.cat([mean_y for _, line_x, mean_y in profiles if mean_y.numel()])
    xlim = padded_limits(all_x, lower_bound=0.0)
    ylim = padded_limits(all_y)

    nrows = len(profiles)
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(6.6, 1.55 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    for row, (dh, line_x, mean_y) in enumerate(profiles):
        ax = axes[row, 0]
        ax.plot(line_x, mean_y, linewidth=1.4, color="C0")
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.6)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_ylabel(f"{perturbation_label(dh, default_dh)}\n{response_cell_type}")

    fig.suptitle(f"{model_name}\n{title}\nresponse = perturbed - baseline activity")
    axes[-1, 0].set_xlabel("Distance from perturbed neuron (um)")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    return fig


def plot_response_strength_distance_orientation_profiles(
    x,
    model_name,
    response_items,
    response_cell_type,
    distance,
    title,
    out,
    dpi,
    default_dh=10000.0,
):
    _, _, _, origin_mask = grid_metadata(x, response_items[0][2])
    cell_types = list(x["cell_type"].categories)
    cell_idx = cell_types.index(response_cell_type)
    ori = orientation_values(x)

    profiles = []
    for dh, dr, perturb_idx in response_items:
        panel = response_panel(dr, cell_idx, perturb_idx)
        profiles.append(
            (
                dh,
                distance_orientation_mean_lines(panel, distance, origin_mask, ori),
            )
        )

    all_x = torch.cat(
        [
            line_x
            for _, lines in profiles
            for _, line_x, mean_y in lines
            if line_x.numel() and mean_y.numel()
        ]
    )
    all_y = torch.cat(
        [
            mean_y
            for _, lines in profiles
            for _, line_x, mean_y in lines
            if line_x.numel() and mean_y.numel()
        ]
    )
    xlim = padded_limits(all_x, lower_bound=0.0)
    ylim = padded_limits(all_y)

    nrows = len(profiles)
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(6.6, 1.55 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    cmap = plt.get_cmap("twilight_shifted")
    norm = colors.Normalize(vmin=float(ori.min()), vmax=float(ori.max()))
    for row, (dh, lines) in enumerate(profiles):
        ax = axes[row, 0]
        for theta, line_x, mean_y in lines:
            ax.plot(line_x, mean_y, linewidth=1.2, color=cmap(norm(theta)))
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.6)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_ylabel(f"{perturbation_label(dh, default_dh)}\n{response_cell_type}")

    fig.suptitle(f"{model_name}\n{title}\nresponse = perturbed - baseline activity")
    axes[-1, 0].set_xlabel("Distance from perturbed neuron (um)")
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        shrink=0.8,
    )
    cbar.set_label("Preferred orientation (deg)")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    return fig


def dh_slug(dh):
    if dh < 0:
        sign = "neg"
    elif dh > 0:
        sign = "pos"
    else:
        sign = "zero"
    magnitude = model_slug(f"{abs(dh):g}")
    return f"dh_{sign}_{magnitude}"


def plot_psi_orientation_profiles(
    x, responses, response_cell_type, perturb_idx, distance, psi, title, out, dpi
):
    _, _, _, origin_mask = grid_metadata(x, perturb_idx)
    cell_types = list(x["cell_type"].categories)
    cell_idx = cell_types.index(response_cell_type)
    distances = selected_distance_values(distance, origin_mask, count=4)
    ori = orientation_values(x)

    summaries = []
    for d in distances:
        per_model = []
        for model_name, dr in responses.items():
            panel = response_panel(dr, cell_idx, perturb_idx)
            lines = [
                (
                    float(theta),
                    *psi_summary_at_distance_orientation(
                        panel, distance, psi, ori_idx, d
                    )[:2],
                )
                for ori_idx, theta in enumerate(ori)
            ]
            per_model.append((model_name, lines))
        summaries.append((d, per_model))

    all_x = torch.cat(
        [
            xs
            for _, per_model in summaries
            for _, lines in per_model
            for _, xs, mean_y in lines
            if xs.numel() and mean_y.numel()
        ]
    )
    all_y = torch.cat(
        [
            mean_y
            for _, per_model in summaries
            for _, lines in per_model
            for _, xs, mean_y in lines
            if xs.numel() and mean_y.numel()
        ]
    )
    xlim = padded_limits(all_x)
    ylim = padded_limits(all_y)

    model_names = list(responses)
    fig, axes = plt.subplots(
        len(summaries),
        len(model_names),
        figsize=(3.3 * len(model_names), 1.75 * len(summaries)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    cmap = plt.get_cmap("twilight_shifted")
    norm = colors.Normalize(vmin=float(ori.min()), vmax=float(ori.max()))
    for row, (d, per_model) in enumerate(summaries):
        for col, (model_name, lines) in enumerate(per_model):
            ax = axes[row, col]
            for theta, xs, mean_y in lines:
                ax.plot(xs, mean_y, linewidth=1.1, color=cmap(norm(theta)))
            ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.6)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            if row == 0:
                ax.set_title(model_name)
            if col == 0:
                ax.set_ylabel(f"d={float(d):g} um\n{response_cell_type}")

    fig.suptitle(f"{title}\nresponse = perturbed - baseline activity")
    for ax in axes[-1, :]:
        ax.set_xlabel("Angle from perturbed neuron, psi (deg)")
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        shrink=0.8,
    )
    cbar.set_label("Preferred orientation (deg)")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    return fig


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

    mean_lines = []
    for model_name, xs, ys in points:
        line_x, mean_y = mean_profile_line(xs, ys, bins=mean_bins)
        mean_lines.append((model_name, line_x, mean_y))
    all_mean_y = torch.cat([line[2] for line in mean_lines if line[2].numel()])
    mean_ylim = padded_limits(all_mean_y)

    mean_ax = axes[-1, 0]
    for n, (model_name, line_x, mean_y) in enumerate(mean_lines):
        color = colors[n % len(colors)]
        mean_ax.plot(line_x, mean_y, linewidth=1.2, label=model_name, color=color)
    mean_ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.6)
    mean_ax.set_ylim(*mean_ylim)
    mean_ax.set_ylabel(f"mean\n{response_cell_type}")
    mean_ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    axes[0, 0].set_title(title)
    axes[-1, 0].set_xlabel(xlabel)
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
    parser.add_argument(
        "--N-space",
        type=int,
        nargs=2,
        default=(50, 50),
        metavar=("N_RADIAL", "N_ANGLE"),
        help="Polar grid resolution: radial samples including origin, angular samples per ring.",
    )
    parser.add_argument("--N-ori", type=int, default=8)
    parser.add_argument(
        "--space-extent",
        type=float,
        default=1000.0,
        help="Diameter of the modeled polar disk in um.",
    )
    parser.add_argument(
        "--response-mode",
        choices=("matrix", "matrix_approx"),
        default="matrix_approx",
    )
    parser.add_argument("--approx-order", type=int, default=8)
    parser.add_argument("--dh", type=float, default=10000.0)
    parser.add_argument(
        "--dh-values",
        type=float,
        nargs="+",
        help=(
            "Perturbation strengths used as profile rows. "
            "Defaults to -2, -1, 0, 1, 2 times --dh."
        ),
    )
    parser.add_argument("--max-neurons", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--experiment-name",
        default="origin_horizontal_perturbation_polar_celltypes",
    )
    parser.add_argument(
        "--only-psi",
        action="store_true",
        help="Only write response-over-psi plots from the polar grid.",
    )
    parser.add_argument(
        "--profiles-only",
        action="store_true",
        help="Write distance and psi profiles, but skip polar spatial and orientation plots.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.dh_values is None:
        args.dh_values = default_dh_values(args.dh)
    validate_args(parser, args)

    fit = args.fit or sorted_fit_paths(args.fits_dir)[args.fit_index]
    state = load_model_state(fit)
    seeds = args.seeds if args.seeds is not None else [args.seed]

    set_rcParams()
    for seed in seeds:
        torch.manual_seed(seed)
        seed_dir = Path("results") / args.experiment_name / f"seed_{seed}"
        write_perturbation_note(seed_dir, args, fit, seed)

        for perturb_cell_type in CELL_TYPES:
            x, _ = make_polar_grid(
                args.N_space, args.N_ori, args.space_extent, CELL_TYPES
            )
            perturb_idx = perturb_origin_horizontal(x, perturb_cell_type, 1.0)
            unit_responses = compute_responses(
                state,
                x,
                seed,
                mode=args.response_mode,
                approx_order=args.approx_order,
            )

            response_items = {model_name: [] for model_name, _ in MODEL_VARIANTS}
            scaled_responses_by_dh = []
            for dh in args.dh_values:
                responses = scale_responses(unit_responses, dh)
                scaled_responses_by_dh.append((dh, responses))
                for model_name, dr in responses.items():
                    response_items[model_name].append((dh, dr, perturb_idx))

            responses = scale_responses(unit_responses, args.dh)
            rel_ori, distance, psi, _ = grid_metadata(x, perturb_idx)

            for response_cell_type in CELL_TYPES:
                if not args.only_psi:
                    if not args.profiles_only:
                        out = (
                            seed_dir
                            / f"perturb_{perturb_cell_type}_response_{response_cell_type}.pdf"
                        )
                        fig = plot_responses_polar(
                            x,
                            responses,
                            response_cell_type,
                            perturb_idx,
                            out,
                            args.dpi,
                        )
                        plt.close(fig)
                        print(f"Saved {out} using fit {fit}.")

                    for model_name, items in response_items.items():
                        out = (
                            seed_dir
                            / (
                                f"perturb_{perturb_cell_type}_response_{response_cell_type}"
                                f"_{model_slug(model_name)}_over_distance.pdf"
                            )
                        )
                        fig = plot_response_strength_distance_profiles(
                            x,
                            model_name,
                            items,
                            response_cell_type,
                            distance,
                            (
                                f"perturb {perturb_cell_type}, {response_cell_type} response, "
                                "response over distance"
                            ),
                            out,
                            args.dpi,
                            args.dh,
                        )
                        plt.close(fig)
                        print(f"Saved {out} using fit {fit}.")

                        out = (
                            seed_dir
                            / (
                                f"perturb_{perturb_cell_type}_response_{response_cell_type}"
                                f"_{model_slug(model_name)}_over_distance_by_orientation.pdf"
                            )
                        )
                        fig = plot_response_strength_distance_orientation_profiles(
                            x,
                            model_name,
                            items,
                            response_cell_type,
                            distance,
                            (
                                f"perturb {perturb_cell_type}, {response_cell_type} response, "
                                "response over distance by orientation preference"
                            ),
                            out,
                            args.dpi,
                            args.dh,
                        )
                        plt.close(fig)
                        print(f"Saved {out} using fit {fit}.")

                    if not args.profiles_only:
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

                for dh, dh_responses in scaled_responses_by_dh:
                    dh_label = perturbation_label(dh, args.dh).replace("\n", ", ")
                    out = (
                        seed_dir
                        / (
                            f"perturb_{perturb_cell_type}_response_{response_cell_type}"
                            f"_{dh_slug(dh)}_over_psi.pdf"
                        )
                    )
                    fig = plot_psi_distance_profiles(
                        x,
                        dh_responses,
                        response_cell_type,
                        perturb_idx,
                        distance,
                        psi,
                        (
                            f"perturb {perturb_cell_type}, {response_cell_type} response, "
                            f"response over psi, {dh_label}"
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
                            f"_{dh_slug(dh)}_over_psi_by_orientation.pdf"
                        )
                    )
                    fig = plot_psi_orientation_profiles(
                        x,
                        dh_responses,
                        response_cell_type,
                        perturb_idx,
                        distance,
                        psi,
                        (
                            f"perturb {perturb_cell_type}, {response_cell_type} response, "
                            f"response over psi by orientation preference, {dh_label}"
                        ),
                        out,
                        args.dpi,
                    )
                    plt.close(fig)
                    print(f"Saved {out} using fit {fit}.")


if __name__ == "__main__":
    main()
