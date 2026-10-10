"""Deterministic acquisition-function maximisation.

Strategy (spec §12): **dense grid + adaptive local refinement**, fully
vectorised (no per-point scipy loop, no random restarts):

* **1D** -- dense grid (``n_grid_1d`` points); the best up to ``n_cells``
  grid cells are each refined in two batched rounds of 101 points.
  Final resolution ~ range / (n_grid_1d * 101 * 101).
* **d >= 2** -- dense Cartesian grid (``n_grid_d`` per dimension); the best
  up to ``n_cells`` non-overlapping grid cells are each subdivided twice
  (``sub``^d points per round, second round around the local best).

All refinement stages evaluate the acquisition in *batches* on the GP
posterior, which is ~an order of magnitude faster than multi-start
L-BFGS-B with numerical gradients (whose Python call overhead dominates in
low dimensions) while converging to the same local maxima for smooth
acquisition functions.  Deterministic for a fixed configuration.
"""

from __future__ import annotations

import numpy as np

from ..base import SearchSpace

__all__ = ["AcquisitionOptimizer"]


class AcquisitionOptimizer:
    def __init__(
        self,
        space: SearchSpace,
        n_grid_1d: int = 401,
        n_grid_d: int = 21,
        n_restarts: int = 3,
        sub: int = 5,
    ) -> None:
        self.space = space
        self.n_grid_1d = int(n_grid_1d)
        self.n_grid_d = int(n_grid_d)
        # kept for interface compatibility: number of refined cells
        self.n_restarts = int(max(n_restarts, 1))
        self.sub = int(max(sub, 3))
        if space.dim > 3:
            raise ValueError("grid-based acquisition optimisation supports d <= 3")

    # ------------------------------------------------------------------ api
    def __call__(self, acq) -> tuple[np.ndarray, float]:
        if self.space.dim == 1:
            return self._optimize_1d(acq)
        return self._optimize_d(acq)

    # --------------------------------------------------------------- helpers
    def _best(self, acq, X: np.ndarray) -> tuple[np.ndarray, float]:
        v = np.atleast_1d(acq(np.atleast_2d(X)))
        i = int(np.argmax(v))
        return np.atleast_2d(X)[i].copy(), float(v[i])

    def _cell_1d(self, x: float, step: float) -> tuple[float, float]:
        lo, hi = float(self.space.lower[0]), float(self.space.upper[0])
        idx = int(np.floor((x - lo) / step + 1e-9))
        a = max(lo, lo + idx * step)
        b = min(hi, a + step)
        return a, b

    # ------------------------------------------------------------------ 1D
    def _optimize_1d(self, acq) -> tuple[np.ndarray, float]:
        lo, hi = float(self.space.lower[0]), float(self.space.upper[0])
        grid = np.linspace(lo, hi, self.n_grid_1d).reshape(-1, 1)
        vals = np.atleast_1d(acq(grid))
        step = (hi - lo) / max(self.n_grid_1d - 1, 1)

        best_x, best_v = grid[int(np.argmax(vals))].copy(), float(vals.max())
        refined_cells: list[tuple[float, float]] = []
        for idx in np.argsort(-vals, kind="stable"):
            idx = int(idx)
            a, b = self._cell_1d(float(grid[idx, 0]), step)
            center = 0.5 * (a + b)
            if any(abs(center - c) < 0.9 * step for c in refined_cells):
                continue
            refined_cells.append(center)

            g1 = np.linspace(a, b, 101).reshape(-1, 1)
            v1 = np.atleast_1d(acq(g1))
            i1 = int(np.argmax(v1))
            a2, b2 = float(g1[max(i1 - 1, 0), 0]), float(g1[min(i1 + 1, 100), 0])
            g2 = np.linspace(a2, b2, 101).reshape(-1, 1)
            v2 = np.atleast_1d(acq(g2))
            i2 = int(np.argmax(v2))
            x, v = float(g2[i2, 0]), float(v2[i2])
            if v > best_v:
                best_x, best_v = np.asarray([x]), v
            if len(refined_cells) >= self.n_restarts:
                break
        return best_x, best_v

    # ------------------------------------------------------------------ d≥2
    def _cell_d(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-dimension cell [a, b] containing the grid point ``x``."""
        n = self.n_grid_d
        w = self.space.range / (n - 1)
        idx = np.floor((x - self.space.lower) / w + 1e-9).astype(int)
        a = np.clip(self.space.lower + idx * w, self.space.lower, self.space.upper)
        b = np.clip(a + w, self.space.lower, self.space.upper)
        return a, b

    @staticmethod
    def _mesh(axes: list[np.ndarray]) -> np.ndarray:
        mesh = np.meshgrid(*axes, indexing="ij")
        return np.stack([m.ravel() for m in mesh], axis=1)

    def _subdivide(self, a: np.ndarray, b: np.ndarray, n_sub: int) -> np.ndarray:
        axes = [np.linspace(a[j], b[j], n_sub) for j in range(len(a))]
        return self._mesh(axes)

    def _optimize_d(self, acq) -> tuple[np.ndarray, float]:
        n = self.n_grid_d
        grid = self.space.grid(n)
        vals = np.atleast_1d(acq(grid))
        best_x, best_v = grid[int(np.argmax(vals))].copy(), float(vals.max())

        refined_centers: list[np.ndarray] = []
        cell_width = self.space.range / (n - 1)
        for flat in np.argsort(-vals, kind="stable"):
            flat = int(flat)
            x = grid[flat]
            a, b = self._cell_d(x)
            center = 0.5 * (a + b)
            if any(np.max(np.abs(center - c)) < 0.9 * np.max(cell_width)
                   for c in refined_centers):
                continue
            refined_centers.append(center)

            # round 1: subdivide the cell
            s1 = self._subdivide(a, b, self.sub)
            v1 = np.atleast_1d(acq(s1))
            i1 = int(np.argmax(v1))
            # round 2: subdivide the sub-cell around the local best
            w2 = (b - a) / self.sub
            idx2 = np.unravel_index(i1, (self.sub,) * self.space.dim)
            a2 = a + np.asarray(idx2) * w2
            b2 = a2 + w2
            s2 = self._subdivide(a2, b2, self.sub)
            v2 = np.atleast_1d(acq(s2))
            i2 = int(np.argmax(v2))
            x2, v2v = s2[i2], float(v2[i2])
            if v2v > best_v:
                best_x, best_v = x2.copy(), v2v
            if len(refined_centers) >= self.n_restarts:
                break
        return best_x, best_v
