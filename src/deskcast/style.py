"""Single matplotlib style helper so every chart shares the same brand look."""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

BG = "#FFFBF6"
INK = "#241A18"
MUTED = "#5E4C46"
GRID = "#EAE0D5"
CRIMSON = "#BE1E3C"
DEEP_CRIMSON = "#8E1630"
GOLD = "#B08A4E"
SAND = "#E3D2B4"

# DejaVu Sans has no semibold face; matplotlib falls back to bold and logs a warning each time.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

FIGSIZE = (10.0, 5.625)  # 1600 x 900 px at dpi=160
DPI = 160


def apply_style() -> None:
    """Apply the brand rcParams (idempotent)."""
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 15,
            "axes.titlelocation": "left",
            "axes.titleweight": "semibold",
            "axes.titlepad": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "axes.prop_cycle": matplotlib.cycler(color=[CRIMSON, GOLD, DEEP_CRIMSON, MUTED, SAND]),
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "legend.frameon": False,
            "font.size": 10.5,
        }
    )


def new_figure():
    """Return (fig, ax) at the standard 1600x900 size."""
    apply_style()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    return fig, ax


def brand_cmap() -> LinearSegmentedColormap:
    """Sequential colormap from background through sand and gold to crimson."""
    return LinearSegmentedColormap.from_list("deskcast", [BG, SAND, GOLD, CRIMSON, DEEP_CRIMSON])


def save(fig, path: str | Path) -> Path:
    """Save at dpi=160 without bbox cropping so the output stays 1600x900."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return path
