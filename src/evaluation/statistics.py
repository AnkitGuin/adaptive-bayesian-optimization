"""Summary statistics and statistical comparisons (spec §31, §32).

* multi-seed summaries (mean / std / median / 95% CI);
* **paired** bootstrap CIs for the difference of two methods (methods are
  paired by seed because all methods share objective/noise trajectories);
* Wilcoxon signed-rank test (paired, exact where feasible);
* Holm-Bonferroni correction for the small families of tests reported per
  study table.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from ..utils.seeds import rng_for

__all__ = [
    "summarize_seeds",
    "paired_bootstrap_diff",
    "wilcoxon_paired",
    "holm_bonferroni",
]


def summarize_seeds(values: np.ndarray) -> dict:
    """values: (n_seeds,) -> dict with mean/std/median/ci95 (t-based)."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)] if v.size else v
    n = v.size
    out = {
        "n_seeds": int(n),
        "mean": float(np.mean(v)) if n else float("nan"),
        "std": float(np.std(v, ddof=1)) if n > 1 else float("nan"),
        "median": float(np.median(v)) if n else float("nan"),
    }
    if n >= 3:
        se = out["std"] / np.sqrt(n)
        tcrit = float(stats.t.ppf(0.975, n - 1))
        out["ci95"] = (out["mean"] - tcrit * se, out["mean"] + tcrit * se)
    elif n == 2:
        se = out["std"] / np.sqrt(2)
        tcrit = float(stats.t.ppf(0.975, 1))
        out["ci95"] = (out["mean"] - tcrit * se, out["mean"] + tcrit * se)
    else:
        out["ci95"] = (float("nan"), float("nan"))
    return out


def paired_bootstrap_diff(a: np.ndarray, b: np.ndarray, n_boot: int = 2000,
                          seed: int = 0) -> dict:
    """Paired bootstrap CI for mean(a) - mean(b).

    ``a`` and ``b`` must be paired by seed (same length).  The CI covers the
    *difference of means*; if it excludes 0 the difference is significant at
    the 95% level.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must be paired (same shape)")
    d = a - b
    n = d.size
    out = {"mean_diff": float(d.mean()), "n": n,
           "ci95": (float("nan"), float("nan")),
           "frac_worse": float((d > 0).mean())}
    if n < 2:
        return out
    rng = rng_for("bootstrap", seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out["ci95"] = (float(lo), float(hi))
    return out


def wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Paired Wilcoxon signed-rank test; returns (statistic, p-value).

    Returns (nan, nan) when fewer than 3 non-zero differences exist (the
    normal approximation / exact tables are not meaningful then; report the
    bootstrap CI instead).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    d = d[np.isfinite(d)]
    d = d[d != 0]
    if d.size < 3:
        return float("nan"), float("nan")
    try:
        res = stats.wilcoxon(d, method="auto")
        return float(res.statistic), float(res.pvalue)
    except ValueError:
        return float("nan"), float("nan")


def holm_bonferroni(pvalues: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (deterministic)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p, kind="stable")
    m = p.size
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj.tolist()
