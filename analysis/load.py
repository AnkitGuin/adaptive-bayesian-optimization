"""Loaders for raw trajectories saved by the experiment scripts.

Raw layout (spec §42):

    results/raw/<study>/<slug>/config.json
    results/raw/<study>/<slug>/summary.json
    results/raw/<study>/<slug>/seed<sid>/<method>.csv
    results/raw/<study>/<slug>/seed<sid>/<method>_drift_events.csv
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["LoadedRun", "load_run", "runs_for_slug", "slug_config", "all_slugs"]


@dataclass
class LoadedRun:
    study: str
    slug: str
    method: str
    seed: int
    df: pd.DataFrame

    # ---------------------------------------------------------------- props
    @property
    def t(self) -> np.ndarray:
        return self.df["t"].to_numpy()

    @property
    def dim(self) -> int:
        # Count decision columns only.  The previous ``filter(like='x') - 1``
        # happened to work in 1D but returned 5 for a 3D trajectory because
        # it also counted all ``xopt_*`` columns.
        return len([c for c in self.df.columns
                    if c.startswith("x") and not c.startswith("xopt_")])

    def x(self) -> np.ndarray:
        cols = [c for c in self.df.columns if c.startswith("x") and not c.startswith("xopt")]
        return self.df[cols].to_numpy(float)

    def x_opt(self) -> np.ndarray:
        cols = [c for c in self.df.columns if c.startswith("xopt_")]
        return self.df[cols].to_numpy(float)

    def f_true(self) -> np.ndarray:
        return self.df["f_true"].to_numpy(float)

    def f_opt(self) -> np.ndarray:
        return self.df["f_opt"].to_numpy(float)

    def regret(self) -> np.ndarray:
        return self.df["regret"].to_numpy(float)

    def cr(self) -> np.ndarray:
        return np.cumsum(self.regret())

    def final_cr(self) -> float:
        return float(self.regret().sum())

    def is_initial(self) -> np.ndarray:
        return self.df["is_initial"].to_numpy(bool)

    def drift_lag(self, t_change: int) -> float:
        """t_detection - t_change of the first detection after the change."""
        ev = self.df[self.df["drift_detected"].to_numpy(bool)]
        ev = ev[ev["t"].to_numpy() >= t_change]
        if ev.empty:
            return float("nan")
        return float(ev["t"].iloc[0] - t_change)


def load_run(study: str, slug: str, method: str, seed: int,
             outdir: Path) -> LoadedRun:
    outdir = Path(outdir)
    df = pd.read_csv(outdir / "raw" / study / slug / f"seed{seed}" / f"{method}.csv")
    return LoadedRun(study=study, slug=slug, method=method, seed=int(seed), df=df)


def slug_config(outdir: Path, study: str, slug: str) -> dict:
    p = Path(outdir) / "raw" / study / slug / "config.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def runs_for_slug(outdir: Path, study: str, slug: str) -> list[LoadedRun]:
    """All (seed, method) runs stored under one slug (sorted, deterministic)."""
    outdir = Path(outdir)
    base = outdir / "raw" / study / slug
    if not base.exists():
        return []
    cfg = slug_config(outdir, study, slug)
    runs = []
    for seed in sorted(int(s) for s in cfg.get("seeds", [])):
        for method in cfg.get("methods", []):
            f = base / f"seed{seed}" / f"{method}.csv"
            if f.exists():
                runs.append(load_run(study, slug, method, seed, outdir))
    return runs


def all_slugs(outdir: Path) -> list[tuple[str, str]]:
    outdir = Path(outdir)
    out = []
    raw = outdir / "raw"
    if not raw.exists():
        return out
    for study_dir in sorted(raw.iterdir()):
        if not study_dir.is_dir():
            continue
        for slug_dir in sorted(study_dir.iterdir()):
            if slug_dir.is_dir():
                out.append((study_dir.name, slug_dir.name))
    return out
