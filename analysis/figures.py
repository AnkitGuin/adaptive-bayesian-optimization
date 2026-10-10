"""Figure generation (spec §36): publication-style PNGs, all computed from
the raw trajectories in ``results/raw``.

Each figure is generated independently; a failure in one (e.g. a skipped
study) never aborts the rest.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.gp import GaussianProcess, WeightedGaussianProcess, make_kernel

from .load import LoadedRun, runs_for_slug, slug_config
from .plot_style import color_for, savefig, style_ax

__all__ = ["make_figures"]


# ---------------------------------------------------------------- helpers
def _first_seed_runs(outdir, study, slug, methods):
    runs = runs_for_slug(outdir, study, slug)
    if not runs:
        return []
    seeds = sorted({r.seed for r in runs})
    keep = []
    for m in methods:
        cand = [r for r in runs if r.method == m and r.seed == seeds[0]]
        if cand:
            keep.append(cand[0])
    return keep


def _cr_curves(runs_by_method: dict[str, list[LoadedRun]]):
    """Per-method mean±std cumulative regret curves (aligned by t)."""
    out = {}
    for m, rs in runs_by_method.items():
        mats = [r.cr().reshape(-1, 1) for r in rs]
        mat = np.concatenate(mats, axis=1)
        out[m] = (mat.mean(axis=1), mat.std(axis=1))
    return out


def _method_runs(outdir, study, slug, methods):
    runs = runs_for_slug(outdir, study, slug)
    return {m: [r for r in runs if r.method == m] for m in methods}


def _rolling(v: np.ndarray, w: int = 5) -> np.ndarray:
    s = pd.Series(v).rolling(w, min_periods=1).mean().to_numpy()
    return s


def _plot_cr(ax, curves, title=None):
    for m, (mean, std) in curves.items():
        c = color_for(m)
        ax.plot(mean, color=c, label=m, linewidth=1.6)
        ax.fill_between(np.arange(len(mean)), mean - std, mean + std,
                        color=c, alpha=0.12, linewidth=0)
    style_ax(ax, title, "evaluation t", "cumulative regret")
    ax.legend(loc="best", ncol=2)


# ---------------------------------------------------------------- figures
def fig01_stationary_landscape(outdir, figdir):
    runs = _first_seed_runs(outdir, "stationary", "stationary-peak-stationary-medium",
                            ["static_ucb"])
    if not runs:
        return None
    r = runs[0]
    n = 30
    X = r.x()[:n]
    y = r.df["y"].to_numpy()[:n]
    cfg = slug_config(outdir, "stationary", "stationary-peak-stationary-medium")
    d = cfg.get("study_defaults", {})
    gp = GaussianProcess(make_kernel(d.get("kernel", "matern52"),
                                     d.get("lengthscale", 0.25)),
                         noise_var=float(d.get("noise_std", 0.05)) ** 2)
    gp.fit(X, y)
    grid = np.linspace(0, 1, 201).reshape(-1, 1)
    mu, var = gp.predict(grid)
    from src.benchmarks import make_benchmark, ScenarioSpec
    from src.utils.seeds import rng_for
    bench = make_benchmark("peak", ScenarioSpec(type="stationary", horizon=n, t_change=50),
                           rng_for("fig01", 0), start=0.3)
    f = bench.evaluate_true(grid, 0)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(grid, f, "k--", alpha=0.6, label=r"$f_t(x)$ (noise-free)")
    ax.plot(grid, mu, color="#d62728", label="GP mean $\\mu(x)$")
    ax.fill_between(grid[:, 0], mu - 2 * np.sqrt(var), mu + 2 * np.sqrt(var),
                    color="#d62728", alpha=0.15, label="$\\mu \\pm 2\\sigma$")
    ax.scatter(X[:n, 0], y, s=14, color="#1f77b4", zorder=5, label="noisy observations")
    ax.axvline(r.x_opt()[0, 0], color="green", ls=":", label=r"$x^*_t$")
    style_ax(ax, "Stationary objective with GP surrogate after 30 evaluations",
             "x", "y")
    ax.legend(loc="lower center", ncol=2)
    return savefig(fig, figdir / "fig01_stationary_landscape.png")


def fig02_moving_optimum(outdir, figdir):
    slug = "drift_slow-peak-slow-medium"
    runs = runs_for_slug(outdir, "drift_slow", slug)
    if not runs:
        return None
    cfg = slug_config(outdir, "drift_slow", slug)
    mu_path = np.array(cfg["benchmark_description"]["mu_path"])
    s = cfg["benchmark_description"].get("peak_sigma", 0.12)
    t_show = [5, 25, 45, 65, 85]
    t_show = [t for t in t_show if t < mu_path.size]
    grid = np.linspace(0, 1, 201)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    cmap = plt.get_cmap("viridis")
    for i, t in enumerate(t_show):
        f = np.exp(-((grid - mu_path[t]) ** 2) / (2 * s * s))
        ax.plot(grid, f, color=cmap(i / max(len(t_show) - 1, 1)),
                label=f"$t={t}$")
        ax.plot(mu_path[t], 1.0, "v", color=cmap(i / max(len(t_show) - 1, 1)), ms=6)
    ax.plot(mu_path[: max(t_show) + 1],
            np.exp(-((mu_path[: max(t_show) + 1] - mu_path[: max(t_show) + 1]) ** 2) / (2 * s * s)),
            "k-", lw=2, alpha=0.8, label=r"$x^*_t$ path")
    style_ax(ax, "Moving optimum: $f_t(x)$ over time", "x", "y")
    ax.legend(loc="lower center", ncol=2)
    return savefig(fig, figdir / "fig02_moving_optimum.png")


def fig03_trajectories(outdir, figdir):
    slug = "drift_slow-peak-slow-medium"
    methods = ["random_search", "static_ucb", "abo", "dabo"]
    runs = _first_seed_runs(outdir, "drift_slow", slug, methods)
    if len(runs) < 2:
        return None
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for r in runs:
        c = color_for(r.method)
        ax.plot(r.t, r.x().ravel(), "-o", ms=2.5, lw=1.0, color=c, alpha=0.85,
                label=r.method)
    xopt = runs[-1].x_opt().ravel()
    ax.plot(runs[-1].t, xopt, "k-", lw=2.2, label=r"true $x^*_t$")
    style_ax(ax, f"Optimization trajectories (slow drift, 1D, seed {runs[0].seed})",
             "iteration t", "x_t")
    ax.legend(loc="best")
    return savefig(fig, figdir / "fig03_trajectories.png")


def _cr_figure(outdir, figdir, name, title, specs):
    import matplotlib.pyplot as plt
    n = len(specs)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), squeeze=False)
    for ax, (study, slug, methods) in zip(axes[0], specs):
        mr = _method_runs(outdir, study, slug, methods)
        curves = _cr_curves({m: rs for m, rs in mr.items() if rs})
        _plot_cr(ax, curves, title=title)
    return savefig(fig, figdir / name)


def fig04_cumulative_regret(outdir, figdir):
    return _cr_figure(
        outdir, figdir, "fig04_cumulative_regret.png",
        "Cumulative regret vs evaluation budget",
        [
            ("drift_slow", "drift_slow-peak-slow-medium",
             ["random_search", "static_ucb", "abo", "dabo"]),
            ("drift_abrupt", "drift_abrupt-peak-abrupt-medium",
             ["random_search", "static_ucb", "static_ei", "abo", "dabo"]),
            ("drift_slow", "drift_slow-peak2d-slow-medium",
             ["random_search", "static_ucb", "abo", "dabo"]),
        ],
    )


def fig05_instantaneous_regret(outdir, figdir):
    import matplotlib.pyplot as plt
    specs = [
        ("drift_slow", "drift_slow-peak-slow-medium",
         ["random_search", "static_ucb", "abo", "dabo"]),
        ("drift_abrupt", "drift_abrupt-peak-abrupt-medium",
         ["random_search", "static_ucb", "abo", "dabo"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), squeeze=False)
    titles = ["slow drift (1D peak)", "abrupt change (1D peak)"]
    for ax, (study, slug, methods), tt in zip(axes[0], specs, titles):
        mr = _method_runs(outdir, study, slug, methods)
        mats = {}
        for m, rs in mr.items():
            if rs:
                mats[m] = np.stack([_rolling(r.regret()) for r in rs], axis=1)
        for m, mat in mats.items():
            c = color_for(m)
            ax.plot(mat.mean(axis=1), color=c, label=m, lw=1.5)
            ax.fill_between(np.arange(mat.shape[0]), mat.mean(axis=1) - mat.std(axis=1),
                            mat.mean(axis=1) + mat.std(axis=1), color=c, alpha=0.10)
        cfg = slug_config(outdir, study, slug)
        tc = cfg.get("benchmark_description", {}).get("scenario", {}).get("t_change")
        if tc is not None and study.startswith("drift_abrupt"):
            ax.axvline(tc, color="k", ls="--", alpha=0.5)
            ax.text(tc + 1, ax.get_ylim()[1] * 0.9, "change", fontsize=8, rotation=90)
        style_ax(ax, tt, "iteration t", "rolling instantaneous regret (w=5)")
        ax.legend(loc="upper right", fontsize=7)
    return savefig(fig, figdir / "fig05_instantaneous_regret.png")


def fig06_objective_vs_time(outdir, figdir):
    slug = "drift_slow-peak-slow-medium"
    methods = ["random_search", "static_ucb", "abo", "dabo"]
    mr = _method_runs(outdir, "drift_slow", slug, methods)
    if not any(mr.values()):
        return None
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for m, rs in mr.items():
        if not rs:
            continue
        mat = np.stack([r.f_true() for r in rs], axis=1)
        c = color_for(m)
        ax.plot(mat.mean(axis=1), color=c, label=m, lw=1.5)
    fo = np.stack([r.f_opt() for r in mr["static_ucb"]], axis=1).mean(axis=1)
    ax.plot(fo, "k-", lw=2, label=r"$f_t(x^*_t)$ (optimum)")
    style_ax(ax, "Objective value vs time (slow drift, 1D)", "iteration t",
             r"noise-free $f_t(x_t)$")
    ax.legend(loc="lower center", ncol=2)
    return savefig(fig, figdir / "fig06_objective_vs_time.png")


def fig07_recovery(outdir, figdir):
    slug = "drift_abrupt-peak-abrupt-medium"
    methods = ["static_ucb", "static_ei", "abo", "dabo"]
    runs = runs_for_slug(outdir, "drift_abrupt", slug)
    if not runs:
        return None
    cfg = slug_config(outdir, "drift_abrupt", slug)
    tc = int(cfg["benchmark_description"]["scenario"]["t_change"])
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    for m in methods:
        rs = [r for r in runs if r.method == m]
        if not rs:
            continue
        mat = np.stack([r.f_true() for r in rs], axis=1)
        c = color_for(m)
        ax.plot(mat.mean(axis=1), color=c, label=m, lw=1.5)
        ax.fill_between(np.arange(mat.shape[0]), mat.mean(axis=1) - mat.std(axis=1),
                        mat.mean(axis=1) + mat.std(axis=1), color=c, alpha=0.10)
    fo = np.stack([r.f_opt() for r in runs], axis=1).mean(axis=1)
    ax.plot(fo, "k--", lw=1.5, alpha=0.6, label=r"$f_t(x^*_t)$")
    ax.axvline(tc, color="k", ls=":", lw=1.5)
    ax.text(tc, ax.get_ylim()[1] * 0.05, f" change t={tc}", fontsize=8)
    # mean detection time for dabo
    dets = []
    for r in runs:
        if r.method != "dabo":
            continue
        ev = r.df[r.df["drift_detected"].to_numpy(bool)]
        ev = ev[ev["t"].to_numpy() >= tc]
        if not ev.empty:
            dets.append(float(ev["t"].iloc[0]))
    if dets:
        td = float(np.mean(dets))
        ax.axvline(td, color=color_for("dabo"), ls="--", lw=1.2)
        ax.text(td + 0.5, ax.get_ylim()[1] * 0.35,
                f"mean detection t={td:.0f}", color=color_for("dabo"), fontsize=8)
    style_ax(ax, "Recovery after abrupt objective change (1D peak)",
             "iteration t", r"noise-free $f_t(x_t)$")
    ax.legend(loc="lower right", fontsize=8)
    return savefig(fig, figdir / "fig07_recovery.png")


def fig08_half_life(outdir, figdir):
    from .tables import find_slugs, _crs_summary
    slugs = find_slugs(outdir, "ablation-peak-abrupt-medium-half_life_")
    if not slugs:
        return None
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    names, means, stds = [], [], []
    order = {"half_life_8": 0, "half_life_16": 1, "half_life_32": 2,
             "half_life_64": 3, "half_life_128": 4, "half_life_inf": 5}
    entries = []
    for _study, slug in slugs:
        val = slug.rsplit("-", 1)[-1]
        runs = runs_for_slug(outdir, "ablation", slug)
        s = _crs_summary(runs)
        entries.append((order.get(val, 9), val, s["mean"], s["std"] if s["n_seeds"] > 1 else 0))
    entries.sort()
    for _, val, mean, std in entries:
        names.append(val)
        means.append(mean)
        stds.append(std)
    xs = np.arange(len(names))
    ax.bar(xs, means, yerr=stds, capsize=4, color="#d62728", alpha=0.85)
    ax.set_xticks(xs, names)
    style_ax(ax, "Effect of forgetting half-life (abrupt change, 1D peak)",
             "half-life (iterations)", "cumulative regret")
    return savefig(fig, figdir / "fig08_half_life.png")


def fig09_noise(outdir, figdir):
    from .tables import find_slugs, _s
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), squeeze=False)
    # (a) stationary 1D peak
    ax = axes[0][0]
    noises = ["low", "medium", "high"]
    xs = np.arange(len(noises))
    w = 0.26
    for i, m in enumerate(("static_ucb", "abo", "dabo")):
        means, stds = [], []
        for nz in noises:
            runs = runs_for_slug(outdir, "stationary", f"stationary-peak-stationary-{nz}")
            s, _ = _s(runs, m)
            means.append(s["mean"])
            stds.append(s["std"] if s["n_seeds"] > 1 else 0)
        ax.bar(xs + (i - 1) * w, means, w, yerr=stds, capsize=3,
               color=color_for(m), label=m)
    ax.set_xticks(xs, noises)
    style_ax(ax, "stationary (1D peak)", "noise level", "cumulative regret")
    ax.legend(fontsize=8)
    # (b) abrupt 1D peak across 4 noise levels
    ax = axes[0][1]
    noises2 = ["low", "medium", "high", "very_high"]
    xs2 = np.arange(len(noises2))
    for i, m in enumerate(("static_ucb", "dabo")):
        means, stds = [], []
        for nz in noises2:
            slugs = find_slugs(outdir, f"sensitivity-peak-abrupt-{nz}-noise_")
            runs = runs_for_slug(outdir, "sensitivity", slugs[0][1]) if slugs else []
            s, _ = _s(runs, m)
            means.append(s["mean"] if s["n_seeds"] else float("nan"))
            stds.append(s["std"] if s["n_seeds"] > 1 else 0)
        ax.bar(xs2 + (i - 0.5) * 0.4, means, 0.4, yerr=stds, capsize=3,
               color=color_for(m), label=m)
    ax.set_xticks(xs2, noises2)
    style_ax(ax, "abrupt change (1D peak)", "noise level", "cumulative regret")
    ax.legend(fontsize=8)
    return savefig(fig, figdir / "fig09_noise.png")


def _fit_history(run: LoadedRun, t_star: int, weighted: bool, half_life: float,
                 noise_std: float, lengthscale: float, kernel_kind: str):
    mask = run.t < t_star
    X = run.x()[mask]
    y = run.df["y"].to_numpy()[mask]
    t_obs = run.t[mask]
    if weighted and (half_life is not None) and np.isfinite(half_life):
        gp = WeightedGaussianProcess(make_kernel(kernel_kind, lengthscale),
                                     base_noise_var=noise_std ** 2)
        gp.fit(X, y, t_obs, t_now=float(t_star), half_life=float(half_life))
    else:
        gp = GaussianProcess(make_kernel(kernel_kind, lengthscale),
                             noise_var=noise_std ** 2)
        gp.fit(X, y)
    return gp


def fig10_acquisition(outdir, figdir):
    slug = "drift_abrupt-peak-abrupt-medium"
    runs = runs_for_slug(outdir, "drift_abrupt", slug)
    if not runs:
        return None
    cfg = slug_config(outdir, "drift_abrupt", slug)
    tc = int(cfg["benchmark_description"]["scenario"]["t_change"])
    d = cfg.get("study_defaults", {})
    noise_std = float(d.get("noise_std", 0.05))
    ls = float(d.get("lengthscale", 0.25))
    kernel_kind = d.get("kernel", "matern52")
    beta_base = float(d.get("beta_base", 2.0))
    beta_high = float(d.get("beta_high", 6.0))
    half_life = float(d.get("half_life", 32.0))
    dabo_cfg = cfg.get("study_defaults", {})
    ls_shrink = 0.5

    def first_seq_after(run, tc):
        # two evaluations after the change: DA-BO has detected (its first
        # post-change residual trips the detector) and is in the recovery
        # window, while static BO is still exploiting the stale optimum.
        seq = run.df[~run.df["is_initial"].to_numpy(bool)]
        after = seq[(seq["t"].to_numpy() >= tc + 2) &
                    (seq["t"].to_numpy() < seq["t"].max() - 2)]
        if after.empty:
            after = seq[seq["t"].to_numpy() >= tc]
        return int(after["t"].iloc[0])

    static_run = next(r for r in runs if r.method == "static_ucb"
                      and r.seed == min(x.seed for x in runs if x.method == "static_ucb"))
    dabo_run = next(r for r in runs if r.method == "dabo"
                    and r.seed == min(x.seed for x in runs if x.method == "dabo"))
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), squeeze=False)
    panels = [
        (axes[0][0], "static_ucb", static_run, False, beta_base, ls, None),
        (axes[0][1], "dabo", dabo_run, True, None, None, half_life),
    ]
    for ax, m, run, weighted, beta_fix, ls_fix, hl in panels:
        t_star = first_seq_after(run, tc)
        in_recovery = False
        if m == "dabo":
            ev = run.df[run.df["drift_detected"].to_numpy(bool)]
            ev = ev[ev["t"].to_numpy() < t_star]
            if not ev.empty and (t_star - int(ev["t"].iloc[-1])) < 20:
                in_recovery = True
        beta = beta_fix if beta_fix is not None else (beta_high if in_recovery else beta_base)
        ls_use = ls_fix if ls_fix is not None else (ls * ls_shrink if in_recovery else ls)
        gp = _fit_history(run, t_star, weighted, hl, noise_std, ls_use, kernel_kind)
        grid = np.linspace(0, 1, 241).reshape(-1, 1)
        mu, var = gp.predict(grid)
        sig = np.sqrt(var)
        f_best = float(run.df["y"].to_numpy()[run.t < t_star].max())
        x_star = float(run.x_opt()[t_star, 0])
        c = color_for(m)
        ax.plot(grid.ravel(), mu, color="#1f77b4", lw=1.8, label=r"$\mu(x)$")
        ax.plot(grid.ravel(), sig, color="#17becf", lw=1.5, ls="--",
                label=r"$\sigma(x)$")
        ax.plot(grid.ravel(), mu + beta * sig, color=c, lw=2.0,
                label=f"UCB $\\mu+{beta:g}\\sigma$")
        ax.axhline(f_best, color="gray", ls=":", lw=1.2)
        ax.text(0.02, f_best + 0.01, "f_best", fontsize=8, color="gray")
        xt_sel = float(run.x()[t_star, 0])
        ax.axvline(xt_sel, color=c, lw=1.4)
        ax.text(xt_sel + 0.01, ax.get_ylim()[0] + 0.05, f"$x_{{t}}$={xt_sel:.2f}",
                color=c, fontsize=8)
        ax.plot([x_star], [1.0], "v", color="green", ms=8)
        ax.text(x_star - 0.02, 0.93, r"$x^*_t$", color="green", fontsize=8)
        tag = " (post-drift boost)" if (m == "dabo" and in_recovery) else ""
        style_ax(ax, f"acquisition at t={t_star}, post-change{tag}", "x", "")
        ax.set_ylim(0, 1.15)
        ax.legend(loc="lower right", fontsize=7)
    return savefig(fig, figdir / "fig10_acquisition.png")


def fig11_ablation(outdir, figdir):
    from .tables import find_slugs, _crs_summary
    variants = ["variant_full", "variant_no_forgetting", "variant_no_drift_detection",
                "variant_no_exploration_boost", "variant_no_memory_shortening"]
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), squeeze=False)
    for ax, bench in zip(axes[0], ("peak", "peak2d")):
        xs, means, stds, labels = [], [], [], []
        for v in variants:
            slugs = find_slugs(outdir, f"ablation-{bench}-abrupt-medium-{v}")
            if not slugs:
                continue
            runs = runs_for_slug(outdir, "ablation", slugs[0][1])
            s = _crs_summary(runs)
            xs.append(len(xs))
            means.append(s["mean"])
            stds.append(s["std"] if s["n_seeds"] > 1 else 0)
            labels.append(v.replace("variant_", ""))
        if xs:
            ax.bar(xs, means, yerr=stds, capsize=4, color="#d62728", alpha=0.85)
            ax.set_xticks(xs, labels, rotation=18, ha="right", fontsize=8)
            style_ax(ax, f"DA-BO ablation, abrupt change ({bench})", "variant",
                     "cumulative regret")
    return savefig(fig, figdir / "fig11_ablation.png")


def fig12_drift_rates(outdir, figdir):
    from .tables import find_slugs, _s
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    # collect (label, static_cr, dabo_cr) per drift regime
    points = []
    slugs = find_slugs(outdir, "sensitivity-peak-rw_")
    sigmas = ("0.004", "0.008", "0.016")
    for _study, slug in sorted(slugs):
        i = slug.split("-rw_")[1].split("-")[0]
        runs = runs_for_slug(outdir, "sensitivity", slug)
        s1, _ = _s(runs, "static_ucb")
        s2, _ = _s(runs, "dabo")
        if i.isdigit() and int(i) < 3:
            points.append((f"walk $\\sigma$={sigmas[int(i)]}", s1["mean"], s2["mean"],
                           s1["std"], s2["std"]))
    runs = runs_for_slug(outdir, "drift_slow", "drift_slow-peak-slow-medium")
    if runs:
        s1, _ = _s(runs, "static_ucb")
        s2, _ = _s(runs, "dabo")
        points.insert(0, ("slow $v$=0.006", s1["mean"], s2["mean"], s1["std"], s2["std"]))
    runs = runs_for_slug(outdir, "drift_abrupt", "drift_abrupt-peak-abrupt-medium")
    if runs:
        s1, _ = _s(runs, "static_ucb")
        s2, _ = _s(runs, "dabo")
        points.append(("abrupt jump", s1["mean"], s2["mean"], s1["std"], s2["std"]))
    if not points:
        return None
    xs = np.arange(len(points))
    ax.errorbar(xs, [p[1] for p in points], yerr=[p[3] for p in points],
                marker="o", color=color_for("static_ucb"), capsize=4, label="static_ucb")
    ax.errorbar(xs, [p[2] for p in points], yerr=[p[4] for p in points],
                marker="s", color=color_for("dabo"), capsize=4, label="dabo")
    ax.set_xticks(xs, [p[0] for p in points], rotation=15, fontsize=8)
    style_ax(ax, "Regime coverage: how drift rate changes the advantage of adaptation",
             "drift regime (increasing change rate)", "cumulative regret")
    ax.legend(fontsize=8)
    return savefig(fig, figdir / "fig12_drift_rates.png")


# ------------------------------------------------------------------ driver
def make_figures(outdir: Path, figdir: Path) -> list[str]:
    outdir, figdir = Path(outdir), Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)
    made = []
    fns = [fig01_stationary_landscape, fig02_moving_optimum, fig03_trajectories,
           fig04_cumulative_regret, fig05_instantaneous_regret,
           fig06_objective_vs_time, fig07_recovery, fig08_half_life,
           fig09_noise, fig10_acquisition, fig11_ablation, fig12_drift_rates]
    for fn in fns:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = fn(outdir, figdir)
            if p:
                made.append(str(p))
                print(f"  figure: {Path(p).name}")
        except Exception as e:  # keep going: one failed figure never blocks the rest
            print(f"  WARNING: {fn.__name__} failed: {type(e).__name__}: {e}")
    return made
