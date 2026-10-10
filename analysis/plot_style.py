"""Consistent publication-style plotting setup (spec §36)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.markersize": 4,
        "legend.fontsize": 8,
        "figure.constrained_layout.use": True,
    }
)

METHOD_COLORS = {
    "random_search": "#999999",
    "static_ei": "#1f77b4",
    "static_ucb": "#ff7f0e",
    "static_pi": "#9467bd",
    "exploitation": "#8c564b",
    "abo": "#2ca02c",
    "dabo": "#d62728",
    "full": "#d62728",
}
METHOD_LINE = {
    "random_search": (0, (2, 2)),
    "static_pi": (0, (4, 2)),
    "exploitation": (0, (1, 1.5)),
}


def color_for(method: str) -> str:
    return METHOD_COLORS.get(method, "#333333")


def style_ax(ax, title: str | None = None, xlabel: str | None = None,
             ylabel: str | None = None) -> None:
    if title:
        ax.set_title(title, fontsize=10)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def savefig(fig, path) -> str:
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)
