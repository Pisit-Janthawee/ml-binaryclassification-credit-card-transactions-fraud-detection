"""Shared plot style + helper to save figures used in the README."""
import matplotlib.pyplot as plt
import seaborn as sns

from .config import FIGURES_DIR


CMAP = "Greys"


def set_style():
    """Grey 'darkgrid' look (same as my Ensemble / Tuberculosis notebooks); red is reserved for fraud."""
    sns.set_style("darkgrid")
    sns.set_palette("gray", 8)
    sns.set_context("notebook", font_scale=1.05)
    plt.rcParams.update({"figure.dpi": 100, "axes.titleweight": "bold"})


def savefig(fig, name: str):
    """Save into reports/figures/<name>.png (these images are committed)."""
    fig.savefig(FIGURES_DIR / f"{name}.png", bbox_inches="tight", dpi=110)
