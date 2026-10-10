"""Ablation study (spec §29, RQ5): remove DA-BO components one at a time and
sweep the forgetting half-life, drift threshold and recovery length.

    python -m experiments.run_ablation [--quick]
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.optimization import StudyDefaults, dabo_ablation

from .common import (
    parse_common_args,
    load_all_configs,
    seed_ids_for,
    run_matrix,
)


def main(argv=None) -> int:
    args = parse_common_args("DA-BO ablation study").parse_args(argv)
    cfg, bench_cfgs, exp_cfg = load_all_configs(args)
    spec = exp_cfg["ablation"]
    quick = args.quick
    if args.seed is not None:
        seeds = [args.seed]
    elif quick:
        seeds = seed_ids_for(cfg, True)
    else:
        # ablations use fewer seeds than the main study
        seeds = seed_ids_for(cfg, False)[: int(cfg.get("seeds.count_ablation", 10))]
    out = Path(args.output)
    noise = spec.get("noise", "medium")

    # ---- component ablations (A-C, plus memory shortening) ----------------
    variants = list(spec["methods"].items())
    for variant, flags in variants:
        def builder(method, d, _flags=flags):
            if method == "full":
                from src.optimization import method_config
                return method_config("dabo", d)
            return dabo_ablation(d, **_flags)

        combos = [(c["benchmark"], c["scenario"]) for c in spec["combos"]]
        run_matrix(
            study="ablation", outdir=out, cfg=cfg, bench_cfgs=bench_cfgs,
            combos=combos, noises=[noise], methods=[variant], seeds=seeds,
            quick=quick, method_builder=builder, label=f"variant_{variant}",
        )

    # ---- parameter sweeps (D, E, F) on the 1D abrupt combo ----------------
    sweep_combo = [(spec["combos"][0]["benchmark"], spec["combos"][0]["scenario"])]

    def sweep_builder(halves):
        def builder(method, d):
            return dabo_ablation(d, half_life=halves)
        return builder

    for h in spec["half_life_sweep"]:
        hh = float("inf") if h == "inf" else float(h)
        run_matrix(
            study="ablation", outdir=out, cfg=cfg, bench_cfgs=bench_cfgs,
            combos=sweep_combo, noises=[noise], methods=[f"half_life_{h}"],
            seeds=seeds, quick=quick, method_builder=sweep_builder(hh),
            label=f"half_life_{h}",
        )
    for thr in spec["threshold_sweep"]:
        run_matrix(
            study="ablation", outdir=out, cfg=cfg, bench_cfgs=bench_cfgs,
            combos=sweep_combo, noises=[noise], methods=[f"threshold_{thr}"],
            seeds=seeds, quick=quick,
            method_builder=lambda m, d, t=thr: dabo_ablation(d, ph_threshold=float(t)),
            label=f"threshold_{thr}",
        )
    for rec in spec["recovery_sweep"]:
        run_matrix(
            study="ablation", outdir=out, cfg=cfg, bench_cfgs=bench_cfgs,
            combos=sweep_combo, noises=[noise], methods=[f"recovery_{rec}"],
            seeds=seeds, quick=quick,
            method_builder=lambda m, d, r=rec: dabo_ablation(d, recovery_len=int(r)),
            label=f"recovery_{rec}",
        )
    print("[ablation] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
