import argparse
import sys
import math
from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from niarb import viz, nn, neurons
from niarb.nn.modules.v1 import compute_osi_scale
from niarb.tensors.circulant import CirculantTensor
from mpl_config import set_rcParams, get_sizes, GREY, GRID_WIDTH


def forward(
    W,
    sigma,
    kappa,
    N_space,
    N_ori,
    N_osi,
    dh,
    mode,
    tau_i=0.5,
    rtol=1e-5,
    maxiter=100,
    output="response",
    **kwargs,
):
    if output not in {"response", "weight"}:
        raise ValueError(f"output must be 'response' or 'weight', but {output=}.")

    if any(Ni % 2 != 0 for Ni in N_space):
        raise ValueError("N_space must be even.")
    if N_ori % 2 != 0:
        raise ValueError("N_ori must be even.")

    N_total = 2 * math.prod(N_space) * (N_ori or 1) * (N_osi or 1)
    if kwargs.get("use_psi") and mode in {"matrix", "matrix_approx", "numerical"}:
        max_neurons = kwargs.pop("max_neurons", None)
        if max_neurons is not None and N_total > max_neurons:
            memory_gib = N_total**2 * 8 / 1024**3
            raise MemoryError(
                f"Psi mode with {mode=} requires a dense {N_total} x {N_total} "
                f"matrix, at least {memory_gib:.1f} GiB for one float64 matrix. "
                f"Reduce --N-space/--N-ori, increase --max-neurons, or run on a "
                f"large-memory compute node."
            )

    sigma = torch.as_tensor(sigma).reshape(2, 2)
    d = len(N_space)

    x = neurons.as_grid(
        2,
        N_space=N_space,
        N_ori=N_ori,
        N_osi=N_osi,
        cell_types=["PYR", "PV"],
        space_extent=(1000,) * d,
    )

    variables = ["cell_type"]
    idx = (0,)
    if N_space:
        variables.append("space")
        idx = idx + tuple(Ni // 2 for Ni in N_space)
        x["distance"] = x["space"].norm(dim=-1)
    if N_ori:
        variables.append("ori")
        idx = idx + (N_ori // 2,)
        x["dori"] = x["ori"].norm(dim=-1)
    if N_osi:
        variables.append("osi")
        idx = idx + (-1,)

    x["dh"] = torch.zeros(x.shape)
    x["dh"][idx] = dh

    model = nn.V1(
        variables,
        cell_types=["PYR", "PV"],
        sigma_optim=False,
        kappa_optim=False,
        tau=[1.0, tau_i],
        mode=mode,
        simulation_kwargs={"dx_rtol": rtol, "options": {"max_num_steps": maxiter}},
        **kwargs,
    )

    model.double()
    x = x.double()

    with torch.inference_mode():
        model.load_state_dict(
            {"sigma": sigma, "gW": W.abs(), "kappa": kappa}, strict=False
        )
        if (abscissa := model.spectral_summary(kind="J").abscissa) >= 0:
            raise ValueError(f"The model is unstable: {abscissa=}")
        print(model.spectral_summary(kind="J"))

        if output == "weight":
            return model(x, ndim=x.ndim, output=output, to_dataframe=False)

        # Independent psi breaks the circulant structure used by the original
        # space/orientation grid solver. The V1 module supports this case as a
        # dense 1D neuron list, while the pandas output below preserves the
        # coordinates needed by the existing plotting code.
        response_ndim = 1 if kwargs.get("use_psi") else x.ndim
        return model(x, ndim=response_ndim, output=output, to_dataframe="pandas")


def plot_ori_response(
    W,
    kappas,
    sigma=(125, 90, 85, 110),
    N_space=(),
    N_ori=12,
    N_osi=0,
    dh=1.0,
    scale=1.0,
    cell_type="PYR",
    **kwargs,
):
    reference_kind = "Matrix" if kwargs.get("use_psi") else "Theory"
    reference_mode = "matrix" if kwargs.get("use_psi") else "analytical"
    dfs = {}
    for kappa in kappas:
        k = compute_osi_scale(torch.distributions.Uniform(0, 1), nn.Identity())
        tw = W.abs() * kappa * k
        if tw[0, 1] * tw[1, 0] - tw[0, 0] * tw[1, 1] > tw[0, 0]:
            category = "Opp.-favoring"
        else:
            category = "Same-favoring"
        print(category)

        df = forward(
            W, sigma, kappa, N_space, N_ori, N_osi, dh * scale, "numerical", **kwargs
        )
        df["dr"] = df["dr"] / scale

        query = f"cell_type == '{cell_type}' and dh == 0"
        dfs[(category, "Simulation")] = df.query(query).reset_index(drop=True)

        df = forward(W, sigma, kappa, (), 50, N_osi, dh, reference_mode, **kwargs)
        df["dr"] = df["dr"] / math.prod(N_space) * (50 / N_ori)
        if N_osi:
            query += "and osi == 0.5"
        dfs[(category, reference_kind)] = df.query(query).reset_index(drop=True)

    df = pd.concat(dfs, names=["category", "kind"]).reset_index([0, 1])

    mapping = {
        "distance": "Distance (μm)",
        "dr": "Response",
        "dori": "Δ tuning pref. (°)",
    }
    lineplot = viz.mapped(sns.lineplot, mapping)

    figsize, rect = get_sizes(1, 1.7, 1, 1)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)
    lineplot(
        df,
        x="dori",
        y="dr",
        hue="category",
        style="kind",
        errorbar=None,
        palette=["#FF766A", "#88CBEC"],
        hue_order=["Same-favoring", "Opp.-favoring"],
        style_order=[reference_kind, "Simulation"],
        ax=ax,
    )
    ax.set_xticks([0, 45, 90])
    viz.remove_legend_subtitles(ax, [2, 2], bbox_to_anchor=(1, 1))

    return fig


def plot_space_response(
    Ws,
    sigma=(40, 40),
    kappa=(0.2, 0.1, 0.1, 0.0),
    N_space=(),
    N_ori=0,
    N_osi=0,
    dh=1.0,
    scale=1.0,
    rlim=(30, 300),
    normalize=False,
    cell_type="PYR",
    **kwargs,
):
    reference_kind = "Matrix" if kwargs.get("use_psi") else "Theory"
    reference_mode = "matrix" if kwargs.get("use_psi") else "analytical"
    if len(sigma) != 2:
        raise ValueError("sigma must be a tuple of length 2.")

    rho = sigma[1] / sigma[0]
    sigma = (*sigma, *sigma)
    kappa = torch.tensor(kappa).reshape(2, 2)

    dfs = {}
    for W in Ws:
        if cell_type == "PYR":
            tr = rho * W[0, 0] + W[1, 1] / rho + rho - 1 / rho
            det = torch.linalg.det(W) + (rho**2 - 1) * W[0, 0]
        else:
            tr = rho * W[0, 0] + W[1, 1] / rho - (rho - 1 / rho)
            det = torch.linalg.det(W) + (rho ** (-2) - 1) * W[1, 1]

        if det > tr**2 / 4:
            category = "∞"
        elif det > 0 and tr < 0:
            category = "1"
        else:
            category = "0"
        print(tr, det, category)

        df = forward(
            W, sigma, kappa, N_space, N_ori, N_osi, dh * scale, "numerical", **kwargs
        )
        df["dr"] = df["dr"] / scale

        query = f"cell_type == '{cell_type}' and dh == 0"
        dfs[(category, "Simulation")] = df.query(query).reset_index(drop=True)

        reference_N_ori = N_ori if kwargs.get("use_psi") else 0
        df = forward(
            W, sigma, kappa, N_space, reference_N_ori, N_osi, dh, reference_mode, **kwargs
        )
        df["dr"] = df["dr"] / ((N_ori or 1) * (N_osi or 1))
        dfs[(category, reference_kind)] = df.query(query).reset_index(drop=True)

    df = pd.concat(dfs, names=["category", "kind"]).reset_index([0, 1])
    df = df.query(f"distance >= {rlim[0]} and distance < {rlim[1]}")
    df = df.reset_index(drop=True)
    if normalize:
        norm = np.zeros(len(df))
        for category, sf in df.groupby("category", observed=True):
            norm[sf.index] = sf.query("kind == @reference_kind")["dr"].max()
        df["dr"] = df["dr"] / norm

    mapping = {
        "distance": "Distance (μm)",
        "dr": "Norm. response" if normalize else "Response",
    }
    lineplot = viz.mapped(sns.lineplot, mapping)

    figsize, rect = get_sizes(1, 1.6, 1, 1)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)
    lineplot(
        df,
        x="distance",
        y="dr",
        hue="category",
        style="kind",
        style_order=[reference_kind, "Simulation"],
        errorbar=None,
        ax=ax,
    )
    ax.axhline(0, color="grey", ls="--")
    viz.remove_legend_subtitles(ax, [3, 2], bbox_to_anchor=(1, 1))

    return fig


def plot_space_ori_response(
    Ws,
    kappas,
    sigma=(100, 100),
    N_space=(),
    N_ori=0,
    N_osi=0,
    dh=1.0,
    scale=1.0,
    rlim=(30, 300),
    normalize=False,
    cell_type="PYR",
    **kwargs,
):
    reference_kind = "Matrix" if kwargs.get("use_psi") else "Theory"
    reference_mode = "matrix" if kwargs.get("use_psi") else "analytical"
    if len(sigma) != 2:
        raise ValueError("sigma must be a tuple of length 2.")

    rho = sigma[1] / sigma[0]
    sigma = (*sigma, *sigma)

    dfs = {}
    for W, kappa in zip(Ws, kappas, strict=True):
        k = compute_osi_scale(torch.distributions.Uniform(0, 1), nn.Identity())
        tW = W * kappa * k
        if cell_type == "PYR":
            tr = rho * tW[0, 0] + tW[1, 1] / rho + rho - 1 / rho
            det = torch.linalg.det(tW) + (rho**2 - 1) * tW[0, 0]
        else:
            tr = rho * tW[0, 0] + tW[1, 1] / rho - (rho - 1 / rho)
            det = torch.linalg.det(tW) + (rho ** (-2) - 1) * tW[1, 1]

        if det > tr**2 / 4:
            category = "∞"
        elif det > 0 and tr < 0:
            category = "1"
        else:
            category = "0"
        print(tr, det, category)

        df = forward(
            W, sigma, kappa, N_space, N_ori, N_osi, dh * scale, "numerical", **kwargs
        )
        df["dr"] = df["dr"] / scale

        query = f"cell_type == '{cell_type}' and dh == 0 and (dori == 0 or dori == 90)"
        dfs[(category, "Simulation")] = df.query(query).reset_index(drop=True)

        df = forward(W, sigma, kappa, N_space, N_ori, N_osi, dh, reference_mode, **kwargs)
        dfs[(category, reference_kind)] = df.query(query).reset_index(drop=True)

    df = pd.concat(dfs, names=["category", "kind"]).reset_index([0, 1])
    df = df.query(f"distance >= {rlim[0]} and distance < {rlim[1]}")
    df = df.reset_index(drop=True)
    if normalize:
        norm = np.zeros(len(df))
        for category, sf in df.groupby("category", observed=True):
            norm[sf.index] = sf.query("kind == @reference_kind")["dr"].max()
        df["dr"] = df["dr"] / norm

    mapping = {
        "distance": "Distance (μm)",
        "dr": "Norm. response" if normalize else "Response",
        "dori": "Δ pref. (°)",
        "category": "# transitions",
    }
    figsize, _ = get_sizes(1, 1, 0.8, 1)
    row_order = [category for category in ["0", "1"] if category in set(df["category"])]
    if not row_order:
        row_order = sorted(df["category"].unique())
    fig, axes = plt.subplots(
        len(row_order),
        1,
        figsize=(figsize[0], figsize[1] * len(row_order)),
        sharex=True,
        squeeze=False,
    )

    for i, category in enumerate(row_order):
        ax = axes[i, 0]
        sf = df.query("category == @category")
        sns.lineplot(
            sf,
            x="distance",
            y="dr",
            hue="dori",
            style="kind",
            style_order=[reference_kind, "Simulation"],
            errorbar=None,
            palette=["#AA2A6C", "#4A489C"],
            legend=(i == 0),
            ax=ax,
        )
        ax.set_title(f"{mapping['category']}: {category}")
        ax.set_xlabel(mapping["distance"] if i == len(row_order) - 1 else "")
        ax.set_ylabel(mapping["dr"])
        if i == 0 and ax.get_legend() is not None:
            sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    fig.tight_layout()
    return fig


def plot_response_comparison(
    kind,
    W,
    kappa,
    sigma=(125, 90, 85, 110),
    N_space=(),
    N_ori=0,
    N_osi=0,
    dh=1.0,
    rlim=(30, 300),
    cell_type="PYR",
    **kwargs,
):
    if kind not in {"space_ori", "ori_osi"}:
        raise ValueError(f"kind must be 'space_ori' or 'ori_osi', but {kind=}.")

    reference_kind = "Matrix" if kwargs.get("use_psi") else "Theory"
    reference_mode = "matrix" if kwargs.get("use_psi") else "analytical"
    df = {}
    df[reference_kind] = forward(
        W, sigma, kappa, N_space, N_ori, N_osi, dh, reference_mode, **kwargs
    )
    df["Numerics"] = forward(
        W, sigma, kappa, N_space, N_ori, N_osi, dh, "numerical", **kwargs
    )
    df["Approx"] = forward(
        W, sigma, kappa, N_space, N_ori, N_osi, dh, "matrix_approx", **kwargs
    )
    df = pd.concat(df, names=["Method"]).reset_index(0)
    df = df.query(f"cell_type == '{cell_type}' and dh == 0")
    df = df.query(f"distance >= {rlim[0]} and distance < {rlim[1]}")

    mapping = {
        "distance": "Distance (μm)",
        "dr": "Response (a.u.)",
        "dori": "Δ Tuning pref. (°)",
        "osi": "Selectivity",
    }
    lineplot = viz.mapped(sns.lineplot, mapping)

    figsize, rect = get_sizes(1, 2, 1, 1)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)

    if kind == "space_ori":
        df = df.query("dori == 0  or dori == 45 or dori == 90")

        lineplot(
            df, x="distance", y="dr", hue="dori", style="Method", errorbar=None, ax=ax
        )
    else:
        df = df.query("osi == 0 or osi == 0.5 or osi == 1")

        lineplot(df, x="dori", y="dr", hue="osi", style="Method", errorbar=None, ax=ax)
        ax.set_xticks([0, 45, 90])

    sns.move_legend(ax, "upper left", bbox_to_anchor=(1.1, 1))
    ax.axhline(0, color="grey", ls="--")

    return fig


def plot_eigvals(
    W,
    kappa,
    sigma=(125, 90, 85, 110),
    N_space=(),
    N_ori=0,
    N_osi=0,
    eps=1e-5,
    **kwargs,
):
    W = forward(
        W,
        sigma,
        kappa,
        N_space,
        N_ori,
        N_osi,
        1.0,
        "numerical",
        output="weight",
        **kwargs,
    )
    if isinstance(W, CirculantTensor):
        W = W.dense(keep_shape=False)
    else:
        ndim = 1 + len(N_space) + bool(N_ori) + bool(N_osi)
        N = 2 * math.prod(N_space) * (N_ori or 1) * (N_osi or 1)
        if W.shape[-2:] != (N, N):
            W = W.reshape((*W.shape[:-2 * ndim], N, N))
    eigvals = torch.linalg.eigvals(W)
    # Exclude very small eigenvalues since there are millions of such eigenvalues
    # which makes Illustrator crash when importing the pdf of this figure
    eigvals = eigvals[eigvals.abs() > eps]
    radius = eigvals.abs().max().item()

    figsize, rect = get_sizes(1, 1, 1, 1, cbar=True)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)

    x = np.linspace(-1, 1, 1000)
    ax.plot(x, np.sqrt(1 - x**2), color=GREY, linewidth=GRID_WIDTH)
    ax.plot(x, -np.sqrt(1 - x**2), color=GREY, linewidth=GRID_WIDTH)
    ax.axvline(1.0, color=GREY, linewidth=GRID_WIDTH, ls="--")
    ax.scatter(eigvals.real, eigvals.imag, s=1, color="black")
    ax.set_xlabel("$\mathrm{Re}(\lambda)$")
    ax.set_ylabel("$\mathrm{Im}(\lambda)$")
    ax.set_xlim(-2.25, 1.75)
    ax.set_xticks([-2, -1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.set_aspect("equal")
    ax.set_title(r"$\mathrm{max}_i |\lambda_i| = %.1f$" % radius)

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", "-m", choices=["space", "ori", "space_ori", "compare", "eigvals"]
    )
    parser.add_argument("--kind", "-k", choices=["space_ori", "ori_osi"])
    parser.add_argument("--wee", type=float, nargs="+")
    parser.add_argument("--wei", type=float, nargs="+")
    parser.add_argument("--wie", type=float, nargs="+")
    parser.add_argument("--wii", type=float, nargs="+")
    parser.add_argument("--kee", type=float, nargs="*", default=())
    parser.add_argument("--kei", type=float, nargs="*", default=())
    parser.add_argument("--kie", type=float, nargs="*", default=())
    parser.add_argument("--kii", type=float, nargs="*", default=())
    parser.add_argument("--N-space", type=int, nargs="*", default=())
    parser.add_argument("--N-ori", type=int, default=0)
    parser.add_argument("--N-osi", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--dh", type=float, default=1.0)
    parser.add_argument("--tau-i", type=float, default=0.5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--approx-order", type=int, default=3)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--rlim", type=float, nargs=2, default=(30, 300))
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--use-psi", action="store_true")
    parser.add_argument(
        "--psi-mode",
        choices=["independent", "visual_field", "direct_space"],
        default="independent",
    )
    parser.add_argument("--use-visual-field-tuning", action="store_true")
    parser.add_argument("--psi", type=float)
    parser.add_argument("--visual-field-map", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--experiment-name", type=str)
    parser.add_argument(
        "--max-neurons",
        type=int,
        default=6000,
        help=(
            "Safety limit for dense psi runs. Set higher only on a large-memory "
            "compute node."
        ),
    )
    parser.add_argument("--dpi", type=int)
    parser.add_argument("--out", "-o", type=str)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    Ws = [
        torch.tensor(W).reshape(2, 2) * torch.tensor([1, -1])
        for W in zip(args.wee, args.wei, args.wie, args.wii, strict=True)
    ]
    kappas = [
        torch.tensor(kappa).reshape(2, 2)
        for kappa in zip(args.kee, args.kei, args.kie, args.kii, strict=True)
    ]

    if args.use_visual_field_tuning:
        if args.psi_mode != "independent":
            parser.error("--use-visual-field-tuning cannot be combined with --psi-mode")
        args.psi_mode = "visual_field"
    use_psi = args.use_psi or args.psi_mode != "independent"
    if use_psi and args.N_ori <= 0:
        parser.error("--use-psi requires --N-ori > 0")
    if args.psi_mode in {"visual_field", "direct_space"} and not args.N_space:
        parser.error(f"--psi-mode {args.psi_mode} requires --N-space")
    if args.psi_mode == "direct_space" and len(args.N_space) != 2:
        parser.error("--psi-mode direct_space requires exactly two --N-space values")
    if args.psi_mode == "visual_field" and not args.visual_field_map:
        parser.error("--psi-mode visual_field requires --visual-field-map")
    if args.psi_mode != "independent" and args.psi is not None:
        parser.error("--psi can only be used with --psi-mode independent")
    if args.seeds is not None and args.seed is not None:
        parser.error("Use either --seed or --seeds, not both")

    model_kwargs_base = {
        "use_psi": use_psi,
        "psi_mode": args.psi_mode,
        "max_neurons": args.max_neurons,
    }
    if args.psi is not None:
        model_kwargs_base["psi"] = args.psi
    if args.visual_field_map:
        model_kwargs_base["visual_field_map"] = (
            "AllenAffineVisualFieldMap",
            {"path": args.visual_field_map},
        )

    if args.seeds is not None:
        seeds = args.seeds
    elif args.seed is not None:
        seeds = [args.seed]
    elif args.experiment_name or args.psi_mode == "direct_space":
        seeds = [0]
    else:
        seeds = [None]

    def make_fig(model_kwargs):
        if args.mode == "ori":
            return plot_ori_response(
                Ws[0],
                kappas,
                N_space=args.N_space,
                N_ori=args.N_ori,
                N_osi=args.N_osi,
                dh=args.dh,
                scale=args.scale,
                tau_i=args.tau_i,
                rtol=args.rtol,
                maxiter=args.maxiter,
                **model_kwargs,
            )
        if args.mode == "space":
            return plot_space_response(
                Ws,
                N_space=args.N_space,
                N_ori=args.N_ori,
                N_osi=args.N_osi,
                dh=args.dh,
                scale=args.scale,
                normalize=args.normalize,
                tau_i=args.tau_i,
                rtol=args.rtol,
                maxiter=args.maxiter,
                rlim=args.rlim,
                **model_kwargs,
            )
        if args.mode == "space_ori":
            return plot_space_ori_response(
                Ws,
                kappas,
                N_space=args.N_space,
                N_ori=args.N_ori,
                N_osi=args.N_osi,
                dh=args.dh,
                scale=args.scale,
                normalize=args.normalize,
                tau_i=args.tau_i,
                rtol=args.rtol,
                maxiter=args.maxiter,
                rlim=args.rlim,
                **model_kwargs,
            )
        if args.mode == "compare":
            return plot_response_comparison(
                args.kind,
                Ws[0],
                kappas[0],
                N_space=args.N_space,
                N_ori=args.N_ori,
                N_osi=args.N_osi,
                dh=args.dh,
                tau_i=args.tau_i,
                rtol=args.rtol,
                maxiter=args.maxiter,
                rlim=args.rlim,
                approx_order=args.approx_order,
                **model_kwargs,
            )
        return plot_eigvals(
            Ws[0],
            kappas[0],
            N_space=args.N_space,
            N_ori=args.N_ori,
            N_osi=args.N_osi,
            tau_i=args.tau_i,
            eps=args.eps,
            **model_kwargs,
        )

    def output_path(seed):
        experiment_name = args.experiment_name
        if experiment_name is None and args.psi_mode == "direct_space":
            experiment_name = "direct_space_psi"
        if experiment_name is None:
            return Path(args.out) if args.out else None

        filename = Path(args.out).name if args.out else f"{args.mode}.pdf"
        seed_label = 0 if seed is None else seed
        out = Path("results") / experiment_name / f"seed_{seed_label}" / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        return out

    set_rcParams()
    if args.dpi is None:
        args.dpi = "figure"
    for seed in seeds:
        if seed is not None:
            torch.manual_seed(seed)
        model_kwargs = dict(model_kwargs_base)
        if seed is not None:
            model_kwargs["seed"] = seed
        fig = make_fig(model_kwargs)
        out = output_path(seed)
        if out is not None:
            fig.savefig(
                out,
                dpi=args.dpi,
                bbox_inches="tight",
                metadata={"Subject": " ".join(sys.argv)},
            )
        if args.show:
            plt.show()
        plt.close(fig)


if __name__ == "__main__":
    main()
