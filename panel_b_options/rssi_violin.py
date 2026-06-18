# -*- coding: utf-8 -*-
"""
Nature-style candidate figures for panel (b): distribution of RSSI per node.
Reuses the SAME forest RSSI model & seeds as rssi_plot.py so data is consistent.
Produces several styles for the user to choose from:
  1) violin            2) raincloud
  3) ridgeline         4) violin + path-loss fit (true log-distance positions)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# ---- fonts: English/numbers -> Times New Roman, Chinese -> SimSun ----------
matplotlib.rcParams["font.family"] = ["Times New Roman", "SimSun"]
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["savefig.dpi"] = 300
matplotlib.rcParams["figure.dpi"] = 150

# ---------------------------------------------------------------- model (same)
nodes = [("Tx1", 1.0), ("Tx2", 35.0), ("Tx3", 60.0), ("Tx4", 100.0), ("Tx5", 150.0)]
RSSI0 = -30.0
dates = {
    "2024-09-01": dict(n=3.05, d_amp=17.0, dew=0.5, shad=1.3, fast=2.2, off=-1.5, seed=901),
    "2024-10-05": dict(n=2.85, d_amp=14.0, dew=0.4, shad=1.1, fast=1.8, off=+1.0, seed=1005),
    "2025-06-15": dict(n=3.00, d_amp=19.0, dew=0.45, shad=1.4, fast=2.4, off=-0.8, seed=615),
}
dt_min = 15
t_h = np.arange(0, 24, dt_min / 60.0)

# Nature-ish muted palette
palette = ["#3B4CC0", "#E8704A", "#2CA02C", "#9467BD", "#17BECF"]


def diurnal_shape(h, dew_strength):
    day = np.cos(2 * np.pi * (h - 13.0) / 24.0)
    dew = np.exp(-((h - 5.0) ** 2) / (2 * 2.5 ** 2))
    return day - dew_strength * dew


def correlated_noise(n, std, corr, rng):
    x = np.zeros(n)
    e = rng.normal(0, std * np.sqrt(1 - corr ** 2), n)
    x[0] = rng.normal(0, std)
    for i in range(1, n):
        x[i] = corr * x[i - 1] + e[i]
    return x


def make_series(dist, p, rng):
    base = RSSI0 - 10 * p["n"] * np.log10(dist) + p["off"]
    dscale = min(1.0, np.log10(dist + 1) / np.log10(151.0))
    shape = diurnal_shape(t_h, p["dew"])
    shape = (shape - shape.min()) / (shape.max() - shape.min()) - 0.5
    diur = p["d_amp"] * dscale * shape
    shadow = correlated_noise(len(t_h), p["shad"] * (0.4 + 0.6 * dscale), 0.85, rng)
    fast = rng.normal(0, p["fast"] * (0.5 + 0.5 * dscale), len(t_h))
    return base + diur + shadow + fast


# pooled samples per node across the 3 dates
data = {name: [] for name, _ in nodes}
for date, p in dates.items():
    rng = np.random.default_rng(p["seed"])
    for name, dist in nodes:
        data[name].append(make_series(dist, p, rng))
for name in data:
    data[name] = np.concatenate(data[name])

samples = [data[name] for name, _ in nodes]
labels = [f"{name}\n{d:g} m" for name, d in nodes]
dists = np.array([d for _, d in nodes])

XL = "\u8282\u70b9 Node"
XLD = "\u8ddd\u79bb Distance /m"
YL = "\u63a5\u6536\u4fe1\u53f7\u5f3a\u5ea6 RSSI /dBm"


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=17)


# =============================================================== 1) VIOLIN
fig, ax = plt.subplots(figsize=(8.2, 6.4))
pos = np.arange(1, len(nodes) + 1)
vp = ax.violinplot(samples, positions=pos, showextrema=False, widths=0.85)
for b, c in zip(vp["bodies"], palette):
    b.set_facecolor(c); b.set_edgecolor("black"); b.set_alpha(0.55); b.set_linewidth(1.0)
# inner boxplot
bp = ax.boxplot(samples, positions=pos, widths=0.12, patch_artist=True,
                showcaps=True, whis=(5, 95),
                medianprops=dict(color="white", linewidth=2),
                boxprops=dict(facecolor="0.25", edgecolor="0.25"),
                whiskerprops=dict(color="0.25"), capprops=dict(color="0.25"),
                flierprops=dict(marker="", alpha=0))
means = [np.mean(s) for s in samples]
ax.scatter(pos, means, marker="o", color="white", edgecolor="black",
           zorder=5, s=42, label="\u5747\u503c Mean")
ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=16)
ax.set_xlabel(XL, fontsize=21); ax.set_ylabel(YL, fontsize=21)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.legend(fontsize=14, loc="upper right", frameon=False)
clean(ax)
fig.tight_layout()
fig.savefig("C:/Users/Administrator/fig_b_violin.png", bbox_inches="tight")
plt.close(fig)
print("saved violin")

# =============================================================== 2) RAINCLOUD
fig, ax = plt.subplots(figsize=(8.6, 6.6))
for i, (s, c) in enumerate(zip(samples, palette)):
    y = len(nodes) - i                       # top-to-bottom Tx1..Tx5
    kde = gaussian_kde(s)
    xs = np.linspace(s.min() - 3, s.max() + 3, 200)
    dens = kde(xs); dens = dens / dens.max() * 0.42
    ax.fill_between(xs, y, y + dens, color=c, alpha=0.6, lw=1.0, edgecolor="black")
    # boxplot below
    q1, med, q3 = np.percentile(s, [25, 50, 75])
    lo, hi = np.percentile(s, [5, 95])
    ax.hlines(y - 0.12, lo, hi, color="0.3", lw=1.4, zorder=3)
    ax.add_patch(plt.Rectangle((q1, y - 0.20), q3 - q1, 0.16,
                 facecolor=c, edgecolor="black", alpha=0.9, zorder=4))
    ax.vlines(med, y - 0.20, y - 0.04, color="white", lw=2, zorder=5)
    # jittered raindrops
    jit = (np.random.default_rng(i).random(len(s)) - 0.5) * 0.14
    ax.scatter(s, np.full_like(s, y - 0.30) + jit, s=5, color=c,
               alpha=0.25, edgecolor="none", zorder=2)
ax.set_yticks([len(nodes) - i for i in range(len(nodes))])
ax.set_yticklabels([f"{n} ({d:g} m)" for n, d in nodes], fontsize=16)
ax.set_xlabel(YL, fontsize=21); ax.set_ylabel(XL, fontsize=21)
ax.grid(axis="x", linestyle="--", alpha=0.35)
clean(ax)
fig.tight_layout()
fig.savefig("C:/Users/Administrator/fig_b_raincloud.png", bbox_inches="tight")
plt.close(fig)
print("saved raincloud")

# =============================================================== 3) RIDGELINE
fig, ax = plt.subplots(figsize=(8.4, 6.8))
overlap = 1.7
for i, (s, c) in enumerate(zip(samples, palette)):
    base = (len(nodes) - 1 - i) * 1.0
    kde = gaussian_kde(s)
    xs = np.linspace(min(x.min() for x in samples) - 3,
                     max(x.max() for x in samples) + 3, 400)
    dens = kde(xs); dens = dens / dens.max() * overlap
    ax.fill_between(xs, base, base + dens, color=c, alpha=0.8,
                    lw=1.2, edgecolor="black", zorder=len(nodes) - i)
    ax.plot(xs, base + dens, color="black", lw=0.8, zorder=len(nodes) - i)
    ax.text(xs[0], base + 0.12, f"{nodes[i][0]} ({nodes[i][1]:g} m)",
            fontsize=15, va="bottom", ha="left")
ax.set_yticks([])
ax.set_xlabel(YL, fontsize=21)
ax.set_ylabel("\u6982\u7387\u5bc6\u5ea6 Probability density (\u504f\u79fb Offset)", fontsize=18)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(labelsize=17)
ax.grid(axis="x", linestyle="--", alpha=0.3)
fig.tight_layout()
fig.savefig("C:/Users/Administrator/fig_b_ridgeline.png", bbox_inches="tight")
plt.close(fig)
print("saved ridgeline")

# =============================================================== 4) VIOLIN + FIT
fig, ax = plt.subplots(figsize=(8.6, 6.6))
logd = np.log10(dists)
# path-loss fit on per-node means
means = np.array([np.mean(s) for s in samples])
A1 = np.vstack([np.ones_like(logd), logd]).T
(A_fit, slope), *_ = np.linalg.lstsq(A1, means, rcond=None)
n_fit = -slope / 10.0
resid = np.concatenate([s - (A_fit + slope * np.log10(d))
                        for s, (_, d) in zip(samples, nodes)])
sigma = np.std(resid)
# x variable is log10(distance); axis is linear in log10(d)
xs = np.linspace(np.log10(0.8), np.log10(185), 200)
yfit = A_fit + slope * xs
ax.plot(xs, yfit, "-", color="0.2", lw=2.2, zorder=1,
        label=f"\u8def\u5f84\u635f\u8017\u62df\u5408 Path-loss fit\n n={n_fit:.2f}, \u03c3={sigma:.1f} dB")
ax.fill_between(xs, yfit - sigma, yfit + sigma,
                color="0.6", alpha=0.2, zorder=0, label="\u00b1\u03c3")
# violins at true log-distance positions
vp = ax.violinplot(samples, positions=logd, widths=0.18, showextrema=False)
for b, c in zip(vp["bodies"], palette):
    b.set_facecolor(c); b.set_edgecolor("black"); b.set_alpha(0.65); b.set_linewidth(1.0)
for d, m, c in zip(logd, means, palette):
    ax.scatter(d, m, color=c, edgecolor="black", s=60, zorder=6)
ax.set_xlim(np.log10(0.8), np.log10(185))
ax.set_xticks(logd)
ax.set_xticklabels([f"{d:g}" for d in dists], fontsize=17)
ax.set_xlabel(XLD, fontsize=21); ax.set_ylabel(YL, fontsize=21)
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.legend(fontsize=13, loc="upper right", frameon=False)
clean(ax)
fig.tight_layout()
fig.savefig("C:/Users/Administrator/fig_b_violin_fit.png", bbox_inches="tight")
plt.close(fig)
print(f"saved violin+fit (n={n_fit:.2f}, sigma={sigma:.2f})")
print("DONE")
