"""Shared figure style so every chart in the notebooks and the paper looks the same."""
import matplotlib as mpl
import matplotlib.pyplot as plt

from .config import FIGURE_DIR

# One fixed colour per model (never re-assigned by rank)
MODEL_COLORS = {
    "HAR-RV": "#2a78d6",
    "log-HAR": "#eb6834",
    "HAR-RV-LSTM": "#1baf7a",
    "EGARCH-LSTM": "#eda100",
    "HAR-EGARCH-LSTM": "#e87ba4",
    "Actual RV": "#0b0b0b",
}
GROUP_COLORS = {"High Liquidity": "#2a78d6", "Medium Liquidity": "#eb6834",
                "Low Liquidity": "#1baf7a"}
SPLIT_SHADES = {"Train": "#f2f2f0", "Validation": "#e6eef9", "Test": "#e8f6ef"}


def set_style():
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "axes.labelcolor": "#52514e", "axes.edgecolor": "#b5b4ae",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#e7e6e2", "grid.linewidth": 0.6,
        "xtick.color": "#52514e", "ytick.color": "#52514e",
        "legend.frameon": False, "lines.linewidth": 1.6,
    })


def save(fig, name):
    """Save to figures/<name>.png and .pdf (the paper uses the PDF)."""
    fig.savefig(FIGURE_DIR / f"{name}.png")
    # PDF copy for the paper: the caption replaces a single-panel title / overall title
    saved = []
    if fig._suptitle is not None:
        saved.append((fig._suptitle, fig._suptitle.get_text()))
        fig._suptitle.set_text("")
    if len(fig.axes) == 1:
        t = fig.axes[0].title
        saved.append((t, t.get_text()))
        t.set_text("")
    fig.savefig(FIGURE_DIR / f"{name}.pdf")
    for obj, text in saved:
        obj.set_text(text)
    return FIGURE_DIR / f"{name}.png"
