"""Regenerate all tables and figures from raw trajectories.

    python -m analysis.make_outputs --output results

Outputs:
    results/tables/table*.csv
    results/figures/fig*.png   (copied to report/figures/)
    results/summary.md
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from .load import all_slugs
from .tables import build_tables, df_to_md
from .figures import make_figures

__all__ = ["main"]

_TABLE_TITLES = {
    "table1_benchmarks": "Table 1 - Benchmark definitions",
    "table2_stationary": "Table 2 - Stationary optimization results (cumulative regret, mean±std)",
    "table3_slow_drift": "Table 3 - Slow-drift results (cumulative regret / tracking error)",
    "table4_abrupt": "Table 4 - Abrupt-shift results (CR, recovery time, adaptation regret)",
    "table5_random_walk": "Table 5 - Random-walk optimum results",
    "table6_noise": "Table 6 - Noise robustness",
    "table7_ablation": "Table 7 - Ablation results",
    "table8_sensitivity": "Table 8 - Sensitivity analysis",
    "table9_application": "Table 9 - Real-world-inspired application",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--benchmarks", default="configs/benchmarks.yaml")
    ap.add_argument("--experiments", default="configs/experiments.yaml")
    ap.add_argument("--output", default="results")
    args = ap.parse_args(argv)

    outdir = Path(args.output)
    figdir = outdir / "figures"
    slugs = all_slugs(outdir)
    if not slugs:
        print(f"no raw results found under {outdir}/raw -- run the experiments first")
        return 1

    print("building tables ...")
    tables = build_tables(outdir)

    print("building figures ...")
    made = make_figures(outdir, figdir)

    # ---------------- Markdown summary ----------------
    md = ["# Results summary (auto-generated)", ""]
    md.append(f"Studies with raw trajectories: {len(slugs)}")
    md.append("")
    for name, df in tables.items():
        md.append(f"## {_TABLE_TITLES.get(name, name)}")
        md.append("")
        md.append(df_to_md(df))
        md.append("")
    (outdir / "summary.md").write_text("\n".join(md), encoding="utf-8")

    # copy figures next to the report
    report_figs = Path("report") / "figures"
    report_figs.mkdir(parents=True, exist_ok=True)
    for p in made:
        shutil.copy(p, report_figs / Path(p).name)

    print(f"wrote {outdir}/summary.md, {len(list((outdir / 'tables').glob('*.csv')))} "
          f"tables, {len(made)} figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
