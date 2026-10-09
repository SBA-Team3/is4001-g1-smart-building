"""
One chart style for every notebook.

Colour follows the entity, never its rank: floor N is always FLOOR_COLORS[N],
AC/lighting/plug are always the same three hues. The categorical order is a
colourblind-validated reference palette (adjacent-pair CVD dE >= 8); for
scatter plots keep to three colours and facet beyond that.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQUENTIAL = "Blues"                         # magnitude: one hue, light -> dark
DIVERGING = "RdBu_r"                         # polarity: blue / gray / red
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

FLOOR_COLORS = {f: CATEGORICAL[f - 1] for f in range(1, 8)}
END_USE_COLORS = {"ac": CATEGORICAL[0], "light": CATEGORICAL[1], "plug": CATEGORICAL[2]}
END_USE_LABELS = {"ac": "Air-conditioning", "light": "Lighting", "plug": "Plug load"}


def setup() -> None:
    """Apply the shared matplotlib style. Call once at the top of a notebook."""
    mpl.rcParams.update({
        "figure.figsize": (10, 4), "figure.dpi": 110, "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE, "axes.edgecolor": AXIS, "axes.labelcolor": INK_2,
        "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.prop_cycle": mpl.cycler(color=CATEGORICAL),
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "lines.linewidth": 2.0, "lines.markersize": 6,
        "legend.frameon": False, "legend.fontsize": 9,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "savefig.bbox": "tight", "savefig.facecolor": SURFACE,
    })


def save(fig, name: str) -> None:
    from .config import FIGURES
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=150)


def floor_legend(ax, floors, **kw):
    handles = [plt.Line2D([], [], color=FLOOR_COLORS[f], lw=2, label=f"F{f}") for f in floors]
    ax.legend(handles=handles, **kw)
