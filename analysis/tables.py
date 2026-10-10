"""Table generation (spec §37): tables 1-9, saved as CSV + Markdown.

All numbers are computed from the raw trajectories in ``results/raw`` --
nothing is hand-entered.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.statistics import (
    holm_bonferroni,
    paired_bootstrap_diff,
    summarize_seeds,
    wilcoxon_paired,
)

from .load import LoadedRun, all_slugs, runs_for_slug, slug_config

__all__ = ["build_tables", "df_to_md"]


# ----------------------------------------------------------------- helpers
def find_slugs(outdir: Path, prefix: str) -> list[tuple[str, str]]:
    return [(s, sl) for s, sl in all_slugs(outdir) if sl.startswith(prefix)]


def _crs(runs: list[LoadedRun], method: str) -> np.ndarray:
    return np.array([r.final_cr() for r in runs if r.method == method], dtype=float)


def _fmt(v: float, std: float | None = None) -> str:
    if std is None or (isinstance(std, float) and not np.isfinite(std)):
        return f"{v:.2f}"
    return f"{v:.2f}\u00b1{std:.2f}"


def _s(runs: list[LoadedRun], method: str, get=lambda r: r.final_cr()):
    vals = np.array([get(r) for r in runs if r.method == method], dtype=float)
    return summarize_seeds(vals), vals


def _recovery(run: LoadedRun, t_change: int, tol_frac: float = 0.05) -> float:
    f_true, f_opt = run.f_true(), run.f_opt()
    n = f_true.size
    for t in range(int(t_change), n):
        tol = max(0.0, tol_frac * float(f_opt[t]))
        if f_true[t] >= f_opt[t] - tol:
            return float(t - t_change)
    return float("nan")


def _adapt_regret(run: LoadedRun, t_change: int, window: int = 30) -> float:
    r = run.regret()
    lo, hi = max(int(t_change), 0), min(int(t_change) + window, r.size)
    return float(r[lo:hi].sum()) if hi > lo else float("nan")


def _t_change(cfg: dict) -> int:
    return int(cfg.get("benchmark_description", {}).get("scenario", {}).get("t_change", 50))


def _tracking(run: LoadedRun) -> float:
    return float(np.linalg.norm(run.x() - run.x_opt(), axis=1).mean())


# ------------------------------------------------------------------ tables
def table1_benchmarks(outdir: Path) -> pd.DataFrame:
    rows = []
    seen = set()
    for study, slug in all_slugs(outdir):
        cfg = slug_config(outdir, study, slug)
        if not cfg:
            continue
        key = (cfg.get("benchmark"), cfg.get("scenario"), cfg.get("noise"))
        if key in seen:
            continue
        seen.add(key)
        desc = cfg.get("benchmark_description", {})
        scen = desc.get("scenario", {})
        rows.append({
            "study": study,
            "benchmark": cfg.get("benchmark"),
            "scenario": cfg.get("scenario"),
            "noise": cfg.get("noise"),
            "dim": desc.get("dim"),
            "domain": str(desc.get("domain")),
            "t_change": scen.get("t_change", "-"),
            "budget": cfg.get("budget"),
            "n_seeds": cfg.get("n_seeds"),
            "methods": len(cfg.get("methods", [])),
        })
    return pd.DataFrame(rows)


def table2_stationary(outdir: Path, noises=("low", "medium", "high")) -> pd.DataFrame:
    rows = []
    for bench in ("peak", "multimodal", "peak2d"):
        for noise in noises:
            slug = f"stationary-{bench}-stationary-{noise}"
            runs = runs_for_slug(outdir, "stationary", slug)
            if not runs:
                continue
            methods = sorted({r.method for r in runs},
                             key=lambda m: ["random_search", "static_ei", "static_ucb",
                                            "static_pi", "exploitation", "abo",
                                            "dabo"].index(m) if m in
                             ["random_search", "static_ei", "static_ucb", "static_pi",
                              "exploitation", "abo", "dabo"] else 99)
            row = {"benchmark": bench, "noise": noise}
            for m in methods:
                s, _ = _s(runs, m)
                row[m] = _fmt(s["mean"], s["std"])
            rows.append(row)
    return pd.DataFrame(rows).set_index(["benchmark", "noise"])


def table3_slow_drift(outdir: Path) -> pd.DataFrame:
    rows = []
    for bench in ("peak", "peak2d"):
        slug = f"drift_slow-{bench}-slow-medium"
        runs = runs_for_slug(outdir, "drift_slow", slug)
        if not runs:
            continue
        cfg = slug_config(outdir, "drift_slow", slug)
        methods = ["random_search", "static_ucb", "abo", "dabo"]
        row = {"benchmark": bench}
        for m in methods:
            s, _ = _s(runs, m)
            st, _ = _s(runs, m, get=_tracking)
            row[f"{m}_CR"] = _fmt(s["mean"], s["std"])
            row[f"{m}_tracking"] = _fmt(st["mean"], st["std"])
        # paired significance dabo vs static_ucb
        a = _crs(runs, "dabo")
        b = _crs(runs, "static_ucb")
        if a.size >= 3:
            boot = paired_bootstrap_diff(a, b, seed=0)
            w, p = wilcoxon_paired(a, b)
            ci = boot["ci95"]
            row["dabo_vs_static_ucb_bootCI95"] = f"({ci[0]:.2f}, {ci[1]:.2f})"
            row["dabo_vs_static_ucb_wilcoxon_p"] = f"{p:.3g}" if np.isfinite(p) else "n/a"
        rows.append(row)
    return pd.DataFrame(rows).set_index("benchmark")


def table4_abrupt(outdir: Path, tol_frac: float = 0.05) -> pd.DataFrame:
    rows = []
    for bench in ("peak", "peak2d"):
        slug = f"drift_abrupt-{bench}-abrupt-medium"
        runs = runs_for_slug(outdir, "drift_abrupt", slug)
        if not runs:
            continue
        t_change = _t_change(slug_config(outdir, "drift_abrupt", slug))
        methods = ["random_search", "static_ei", "static_ucb", "abo", "dabo"]
        for m in methods:
            s, _ = _s(runs, m)
            rec = np.array([_recovery(r, t_change, tol_frac) for r in runs if r.method == m])
            rec = rec[np.isfinite(rec)]
            ar = np.array([_adapt_regret(r, t_change) for r in runs if r.method == m])
            lag = np.array([r.drift_lag(t_change) for r in runs if r.method == m])
            lag = lag[np.isfinite(lag)]
            n_det = int((~np.isnan(np.array(
                [r.drift_lag(t_change) for r in runs if r.method == m]))).sum())
            row = {"benchmark": bench, "method": m,
                   "CR_mean±std": _fmt(s["mean"], s["std"]),
                   "recovery_mean": f"{np.mean(rec):.1f}" if rec.size else "n/a",
                   "recovery_median": f"{np.median(rec):.1f}" if rec.size else "n/a",
                   "adapt_regret_30": _fmt(float(np.mean(ar))),
                   "detections": f"{n_det}/{s['n_seeds']}",
                   "detection_lag_mean": f"{np.mean(lag):.1f}" if lag.size else "n/a"}
            rows.append(row)
    return pd.DataFrame(rows).set_index(["benchmark", "method"])


def table5_random_walk(outdir: Path) -> pd.DataFrame:
    slug = "drift_random_walk-peak-random_walk-medium"
    runs = runs_for_slug(outdir, "drift_random_walk", slug)
    if not runs:
        return pd.DataFrame()
    methods = ["random_search", "static_ucb", "abo", "dabo"]
    rows = []
    for m in methods:
        s, _ = _s(runs, m)
        st, _ = _s(runs, m, get=_tracking)
        rows.append({"method": m, "CR_mean±std": _fmt(s["mean"], s["std"]),
                     "tracking_mean±std": _fmt(st["mean"], st["std"])})
    return pd.DataFrame(rows).set_index("method")


def table6_noise(outdir: Path) -> pd.DataFrame:
    rows = []
    # stationary 1D peak (from the stationary study)
    for noise in ("low", "medium", "high"):
        runs = runs_for_slug(outdir, "stationary", f"stationary-peak-stationary-{noise}")
        if not runs:
            continue
        row = {"regime": "stationary", "noise": noise}
        for m in ("static_ucb", "abo", "dabo"):
            s, _ = _s(runs, m)
            row[m] = _fmt(s["mean"], s["std"])
        rows.append(row)
    # abrupt 1D peak across 4 noise levels (from the sensitivity noise sweep)
    for noise in ("low", "medium", "high", "very_high"):
        slugs = find_slugs(outdir, f"sensitivity-peak-abrupt-{noise}-noise_")
        if not slugs:
            continue
        runs = runs_for_slug(outdir, "sensitivity", slugs[0][1])
        row = {"regime": "abrupt", "noise": noise}
        for m in ("static_ucb", "dabo"):
            s, _ = _s(runs, m)
            row[m] = _fmt(s["mean"], s["std"])
        rows.append(row)
    return pd.DataFrame(rows).set_index(["regime", "noise"])


def table7_ablation(outdir: Path) -> pd.DataFrame:
    rows = []
    for bench in ("peak", "peak2d"):
        prefix = f"ablation-{bench}-abrupt-medium-variant_"
        slugs = find_slugs(outdir, prefix)
        if not slugs:
            continue
        base = None
        entries = []
        for _study, slug in slugs:
            variant = slug.rsplit("-", 1)[-1]
            runs = runs_for_slug(outdir, "ablation", slug)
            s = _crs_summary(runs)  # each variant slug holds exactly one method
            entries.append((variant, s["mean"], s["std"], s["n_seeds"]))
            if variant == "variant_full":
                base = s["mean"]
        for variant, mean, std, n in entries:
            delta = "" if base is None or variant == "variant_full" else f"{100 * (mean - base) / max(abs(base), 1e-9):+.1f}%"
            rows.append({"benchmark": bench, "variant": variant,
                         "CR_mean±std": _fmt(mean, std), "delta_vs_full": delta})
    # parameter sweeps (1D abrupt only); val is e.g. "half_life_16"
    for family, key in (("half-life", "half_life"), ("PH threshold", "threshold"),
                        ("recovery length", "recovery")):
        for _study, slug in find_slugs(outdir, f"ablation-peak-abrupt-medium-{key}_"):
            val = slug.rsplit("-", 1)[-1].split(f"{key}_", 1)[-1]
            runs = runs_for_slug(outdir, "ablation", slug)
            s = _crs_summary(runs)
            rows.append({"benchmark": "peak", "variant": f"{family} = {val}",
                         "CR_mean±std": _fmt(s["mean"], s["std"]), "delta_vs_full": ""})
    return pd.DataFrame(rows).set_index(["benchmark", "variant"])


def _crs_summary(runs: list[LoadedRun]) -> dict:
    vals = np.array([r.final_cr() for r in runs], dtype=float)
    return summarize_seeds(vals)


def table8_sensitivity(outdir: Path) -> pd.DataFrame:
    rows = []

    def add(param, value, runs, methods=("static_ucb", "dabo")):
        row = {"parameter": param, "value": value}
        for m in methods:
            s, _ = _s(runs, m)
            row[m] = _fmt(s["mean"], s["std"]) if s["n_seeds"] else "n/a"
        rows.append(row)

    for _study, slug in find_slugs(outdir, "sensitivity-peak-slow-medium-beta_"):
        runs = runs_for_slug(outdir, "sensitivity", slug)
        add("exploration_beta (slow drift 1D)", slug.rsplit("-", 1)[-1], runs)
    for _study, slug in find_slugs(outdir, "sensitivity-peak-abrupt-medium-budget_"):
        runs = runs_for_slug(outdir, "sensitivity", slug)
        add("budget (abrupt 1D)", slug.rsplit("-", 1)[-1], runs)
    for _study, slug in find_slugs(outdir, "sensitivity-peak2d-abrupt-medium-budget_"):
        runs = runs_for_slug(outdir, "sensitivity", slug)
        add("budget (abrupt 2D)", slug.rsplit("-", 1)[-1], runs)
    for _study, slug in find_slugs(outdir, "sensitivity-peak-slow-medium-ninitial_"):
        runs = runs_for_slug(outdir, "sensitivity", slug)
        add("initial_design_size (slow drift 1D)", slug.rsplit("-", 1)[-1], runs)
    for _study, slug in find_slugs(outdir, "sensitivity-peak-abrupt-"):
        if "-noise_" not in slug:
            continue
        runs = runs_for_slug(outdir, "sensitivity", slug)
        noise = slug.split("-abrupt-")[1].split("-noise_")[0]
        add("noise (abrupt 1D)", noise, runs)
    for _study, slug in find_slugs(outdir, "sensitivity-peak-rw_"):
        runs = runs_for_slug(outdir, "sensitivity", slug)
        i = slug.split("-rw_")[1].split("-")[0]
        sigmas = ("0.004", "0.008", "0.016")
        add("random_walk_sigma (1D peak)", sigmas[int(i)] if i.isdigit() and int(i) < 3 else i, runs)
    for _study, slug in find_slugs(outdir, "sensitivity-peak-slow-medium-kernel_"):
        runs = runs_for_slug(outdir, "sensitivity", slug)
        add("kernel (slow drift 1D)", slug.rsplit("-", 1)[-1], runs,
            methods=("static_ucb", "abo", "dabo"))
    return pd.DataFrame(rows)


def table9_application(outdir: Path) -> pd.DataFrame:
    slugs = find_slugs(outdir, "application-process-")
    if not slugs:
        return pd.DataFrame()
    runs = runs_for_slug(outdir, "application", slugs[0][1])
    rows = []
    methods = ["random_search", "static_ei", "static_ucb", "abo", "dabo"]
    for m in methods:
        s, _ = _s(runs, m)
        fin, _ = _s(runs, m, get=lambda r: r.f_true()[-1])
        trk, _ = _s(runs, m, get=_tracking)
        rows.append({"method": m, "CR_mean±std": _fmt(s["mean"], s["std"]),
                     "final_efficiency": _fmt(fin["mean"], fin["std"]),
                     "tracking_error": _fmt(trk["mean"], trk["std"])})
    return pd.DataFrame(rows).set_index("method")


def build_tables(outdir: Path) -> dict[str, pd.DataFrame]:
    out = {
        "table1_benchmarks": table1_benchmarks(outdir),
        "table2_stationary": table2_stationary(outdir),
        "table3_slow_drift": table3_slow_drift(outdir),
        "table4_abrupt": table4_abrupt(outdir),
        "table5_random_walk": table5_random_walk(outdir),
        "table6_noise": table6_noise(outdir),
        "table7_ablation": table7_ablation(outdir),
        "table8_sensitivity": table8_sensitivity(outdir),
        "table9_application": table9_application(outdir),
    }
    outdir = Path(outdir)
    tdir = outdir / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        if df is not None and not df.empty:
            df.to_csv(tdir / f"{name}.csv")
    return out


# --------------------------------------------------------------- markdown
def df_to_md(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_(no data)_\n"
    df = df.copy()
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    else:
        df = df.reset_index()
    df = df.fillna("n/a")
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    body = "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)
    )
    return "\n".join([header, sep, body]) + "\n"
