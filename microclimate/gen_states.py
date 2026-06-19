# -*- coding: utf-8 -*-
"""Generate envelope-PDF fitting figures for high / low / representative states.

All use the same normalized envelope r = sqrt(P / mean(P))  (Eq. 2.3) and the
unified project style.  Reuses helpers from rssi_fading.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rssi_fading import (rssi_state, envelope, fit_all, style,
                         C_GAUSS, C_RICE, C_NAKA, C_RAYL, C_HIST, OUT)


def make(state, title, fname, nbins=42):
    r = envelope(rssi_state[state])
    fits, centers, counts, edges = fit_all(r, nbins=nbins)
    xg = np.linspace(r.min(), r.max(), 400)
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    ax.bar(centers, counts, width=(edges[1] - edges[0]) * 0.95, color=C_HIST,
           edgecolor="#5B6E8C", linewidth=0.5, alpha=0.75,
           label="\u5b9e\u6d4b\u6570\u636e Measured", zorder=2)
    ax.plot(xg, fits["Gaussian"]["pdf"](xg), color=C_GAUSS, lw=2.6, ls="-",
            label="\u9ad8\u65af\u5206\u5e03 Gaussian", zorder=5)
    ax.plot(xg, fits["Rician"]["pdf"](xg), color=C_RICE, lw=2.6, ls=(0, (1, 1)),
            label="Rician \u5206\u5e03", zorder=5)
    ax.plot(xg, fits["Nakagami"]["pdf"](xg), color=C_NAKA, lw=2.6, ls=(0, (6, 3)),
            label="Nakagami \u5206\u5e03", zorder=5)
    ax.plot(xg, fits["Rayleigh"]["pdf"](xg), color=C_RAYL, lw=2.4, ls="-",
            marker="x", markevery=22, markersize=9, markeredgewidth=2,
            label="Rayleigh \u5206\u5e03", zorder=4)
    ax.set_xlabel("\u5f52\u4e00\u5316\u5305\u7edc Normalized envelope", fontsize=22)
    ax.set_ylabel("\u6982\u7387\u5bc6\u5ea6 Probability density", fontsize=22)
    ax.set_xlim(0, r.max())
    ax.set_ylim(0, None)
    ax.legend(fontsize=15, framealpha=0.95, loc="upper right")
    ax.set_title(title, fontsize=16, pad=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    style(ax, fs=18)
    fig.tight_layout()
    fig.savefig(OUT + fname, bbox_inches="tight")
    plt.close(fig)
    print(fname, {k: round(fits[k]["Erms"], 4) for k in fits})


# high-fading (wet veg-soil + most humid air, lowest VPD) -> Rayleigh-like
make(("\u9ad8 High", "\u4f4e Low"),
     "\u9ad8\u6e7f\u5f3a\u8870\u843d\u72b6\u6001  High humidity (strong fading)",
     "fig_fading_pdf_high.png")
# low-fading (dry veg-soil + driest air, highest VPD) -> Gaussian/Rician
make(("\u4f4e Low", "\u9ad8 High"),
     "\u5e72\u71e5\u7a33\u5b9a\u72b6\u6001  Dry / stable (weak fading)",
     "fig_fading_pdf_low.png")
# representative moderate-fading link (already the official fig_fading_pdf.png)
make(("\u4e2d Mid", "\u4e2d Mid"),
     "\u4ee3\u8868\u6027\u94fe\u8def\uff08\u4e2d\u7b49\u8870\u843d\uff09  Representative link (moderate fading)",
     "fig_fading_pdf.png")
