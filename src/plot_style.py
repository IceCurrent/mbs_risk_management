"""Shared matplotlib style for the notebooks' static figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "artifacts" / "figures"

# Categorical slots in fixed order (validated for colour-vision deficiency as
# a set of three). Pools always map to the same slot.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SEQUENTIAL = ("#cde2fb", "#86b6ef", "#2a78d6", "#184f95")
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
REFERENCE = "#8a8984"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"


def apply() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "font.size": 9.5,
            "lines.linewidth": 2.0,
            "lines.markersize": 5,
            "legend.frameon": False,
            "figure.dpi": 110,
        }
    )


def save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path
