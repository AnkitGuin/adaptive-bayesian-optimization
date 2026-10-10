"""Real-world-inspired application (spec §34): 3D process-efficiency
simulator under equipment degradation + demand cycle.

    python -m experiments.run_application [--quick]
"""

from __future__ import annotations

import sys
from pathlib import Path

from .common import (
    parse_common_args,
    load_all_configs,
    run_matrix,
)


def main(argv=None) -> int:
    args = parse_common_args("process-optimization application study").parse_args(argv)
    cfg, bench_cfgs, exp_cfg = load_all_configs(args)
    spec = exp_cfg["application"]
    quick = args.quick
    if args.seed is not None:
        seeds = [args.seed]
    else:
        n = 3 if quick else int(spec.get("seeds", 10))
        base = int(cfg.get("random.base_seed", 20260923))
        seeds = [base + 1000 + i for i in range(n)]  # offset: independent of main study

    out = Path(args.output)
    run_matrix(
        study="application", outdir=out, cfg=cfg, bench_cfgs=bench_cfgs,
        combos=[(spec["benchmark"], "degradation")],
        noises=["sim"],
        methods=list(spec["methods"]),
        seeds=seeds,
        quick=quick,
        label="",
        budget_override=None if quick else int(spec.get("budget", 80)),
        noise_std_override=float(spec.get("noise_std", 1.0)),
        extra_defaults={
            "n_initial": 8 if quick else int(spec.get("n_initial", 12)),
            "noise_std": float(spec.get("noise_std", 1.0)),
            "n_grid_d": 15 if quick else 9,
            "n_restarts": 3,
        },
    )
    print(f"[application] done. (simulator noise std = {spec.get('noise_std', 1.0)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
