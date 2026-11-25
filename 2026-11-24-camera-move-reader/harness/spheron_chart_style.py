"""Spheron brand style for matplotlib study charts.

Matches the landing site's research charts (components/Research/Charts.tsx) and BRAND.md:
black panel, hairline grid, white and sky ink, brand blue only for the series the finding is
about, a distinct marker per series so identity never rests on colour, square corners, a 1px
hairline frame with four corner registration brackets, SF Pro for text, OCR A Std for the small
uppercase mono tag. No glows, no shadows, no purple or warm accents.

Fonts: SF Pro and OCR A Std are not redistributed here. Point SPHERON_FONT_DIR at a folder
holding SFPRODISPLAYREGULAR.otf and OCRAStd.otf to use them; otherwise the chart falls back to
the system sans and mono faces with the same colours and layout.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

SURFACE = "#050608"          # --v2-panel
INK = (1, 1, 1, 0.92)
INK_MUTED = (1, 1, 1, 0.55)  # --v2-muted
INK_DIM = (1, 1, 1, 0.38)    # --v2-dim
GRID = (1, 1, 1, 0.08)       # --v2-line
AXIS = (1, 1, 1, 0.22)
BRACKET = (1, 1, 1, 0.45 * 0.6)
BLUE = "#2F7BFF"             # rgb(47,123,255), the series a finding is about
SKY = "#8FD8FF"
WHITE = "#FFFFFF"
FADED = (1, 1, 1, 0.5)

# role -> (colour, marker). Same four roles as the site's SERIES table.
ROLE = {
    "finding": (BLUE, "o"),
    "second": (SKY, "s"),
    "reference": (WHITE, "D"),
    "other": (FADED, "^"),
}

SANS, MONO = "sans-serif", "monospace"


def setup():
    """Register brand fonts if available and set rcParams. Call once before drawing."""
    global SANS, MONO
    d = os.environ.get("SPHERON_FONT_DIR")
    if d:
        for f, slot in (("SFPRODISPLAYREGULAR.otf", "sans"), ("OCRAStd.otf", "mono")):
            p = os.path.join(d, f)
            if os.path.exists(p):
                font_manager.fontManager.addfont(p)
                name = font_manager.FontProperties(fname=p).get_name()
                if slot == "sans":
                    SANS = name
                else:
                    MONO = name
    plt.rcParams.update({
        "font.family": [SANS, "Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK_MUTED, "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_MUTED, "ytick.labelcolor": INK_MUTED,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 6, "ytick.major.pad": 6,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "legend.frameon": False, "legend.labelcolor": INK_MUTED,
        "svg.fonttype": "path",
    })


def frame(fig, tag=None, title=None, subtitle=None):
    """Hairline border, corner brackets, mono tag and left-aligned title on a figure."""
    w, h = fig.get_size_inches() * fig.dpi
    pad = 6 / w, 6 / h
    bx, by = 14 / w, 14 / h  # bracket arm length
    lo, hi = (pad[0], pad[1]), (1 - pad[0], 1 - pad[1])
    fig.add_artist(plt.Rectangle(lo, hi[0] - lo[0], hi[1] - lo[1], transform=fig.transFigure,
                                 fill=False, edgecolor=GRID, linewidth=1))
    for (x, y, sx, sy) in ((lo[0], lo[1], 1, 1), (hi[0], lo[1], -1, 1), (lo[0], hi[1], 1, -1), (hi[0], hi[1], -1, -1)):
        x2, y2 = x + sx * 4 / w, y + sy * 4 / h  # brackets sit just inside the border
        fig.add_artist(Line2D([x2, x2 + sx * bx], [y2, y2], transform=fig.transFigure, color=BRACKET, linewidth=1.2))
        fig.add_artist(Line2D([x2, x2], [y2, y2 + sy * by], transform=fig.transFigure, color=BRACKET, linewidth=1.2))
    top = 1 - 30 / h
    if tag:
        fig.text(28 / w, top, tag.upper(), fontfamily=[MONO, "Menlo", "DejaVu Sans Mono"], fontsize=8,
                 color=INK_DIM, va="top")
        top -= 20 / h
    if title:
        fig.text(28 / w, top, title, fontsize=13, color=INK, va="top")
        top -= 22 / h
    if subtitle:
        fig.text(28 / w, top, subtitle, fontsize=9.5, color=INK_MUTED, va="top")


def style_ax(ax, zero=None):
    ax.tick_params(axis="both", labelsize=9)
    if zero is not None:
        ax.axhline(zero, color=AXIS, linewidth=1, zorder=1)


def point(ax, x, y, lo=None, hi=None, role="other", marker=None, size=7.5):
    """A dot with an optional interval, outlined in the panel colour like the site's markers."""
    c, m = ROLE[role]
    m = marker or m
    if m == "^":
        size += 2  # a triangle reads smaller than a circle at the same size
    if lo is not None:
        ax.plot([x, x], [lo, hi], color=c, linewidth=1.5, zorder=2, solid_capstyle="butt")
        for yy in (lo, hi):
            ax.plot([x - 0.06, x + 0.06], [yy, yy], color=c, linewidth=1.5, zorder=2)
    ax.plot([x], [y], marker=m, markersize=size, color=c, markeredgecolor=SURFACE, markeredgewidth=1.8,
            linestyle="none", zorder=3)


def legend_handles(items):
    """items: [(label, role, marker or None)] -> handles for fig.legend."""
    out = []
    for label, role, marker in items:
        c, m = ROLE[role]
        out.append(Line2D([], [], color=c, marker=marker or m, markersize=6.5, markeredgecolor=SURFACE,
                          markeredgewidth=1.5, linestyle="none", label=label))
    return out


def save(fig, path_stem):
    for ext in ("svg", "png"):
        fig.savefig(f"{path_stem}.{ext}", dpi=160)
    plt.close(fig)
