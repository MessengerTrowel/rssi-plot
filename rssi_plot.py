# -*- coding: utf-8 -*-
"""
Generate full-day RSSI time-series for 5 transmit nodes (Tx1..Tx5) at distances
1 / 35 / 60 / 100 / 150 m from the receiver Rx, for three dates:
2024-09-01, 2024-10-05, 2025-06-15.

Model (physically motivated, realistic):
  RSSI(d, t) = RSSI0 - 10*n*log10(d)            # log-distance path loss
               + diurnal(t, d, date)            # slow daily (temp + dew) cycle
               + shadowing(t)                    # correlated slow shadow fading
               + fast_fading(t)                  # fast multipath noise
The diurnal amplitude grows with distance (more vegetated/atmospheric path),
and depends on season (foliage / heat / humidity).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# Mixed fonts: English/numbers -> Times New Roman, Chinese -> SimSun (宋体).
# matplotlib (>=3.6) does per-glyph font fallback across this list.
matplotlib.rcParams["font.family"] = ["Times New Roman", "SimSun"]
matplotlib.rcParams["font.serif"] = ["Times New Roman", "SimSun"]
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.size"] = 25

# ---------------------------------------------------------------- node config
nodes = [
    ("Tx1", 1.0),
    ("Tx2", 35.0),
    ("Tx3", 60.0),
    ("Tx4", 100.0),
    ("Tx5", 150.0),
]
# style like the reference figure: distinct color + marker per curve
styles = [
    ("#000000", "s"),   # Tx1 black square
    ("#e8000b", "o"),   # Tx2 red circle
    ("#1f4ed8", "^"),   # Tx3 blue up-triangle
    ("#8a2be2", "v"),   # Tx4 purple down-triangle
    ("#00a0a0", "D"),   # Tx5 teal diamond
]

RSSI0 = -30.0          # reference RSSI at 1 m (dBm)

# per-date environmental parameters
# n      : path-loss exponent (foliage density)
# d_amp  : diurnal swing scale (dB) at far node
# dew    : strength of pre-dawn dew attenuation
# shad   : shadow-fading std (dB)
# fast   : fast-fading std (dB)
# off    : overall offset (dB) capturing seasonal humidity/leaf water
# d_amp is the peak-to-peak smooth diurnal swing (dB) at the far node.
# Forest medium => strong day-night cycle: lower at night, higher in daytime.
dates = {
    "2024-09-01": dict(n=3.05, d_amp=17.0, dew=0.5, shad=2.2, fast=3.8, off=-1.5, seed=901),
    "2024-10-05": dict(n=2.85, d_amp=14.0, dew=0.4, shad=1.9, fast=3.2, off=+1.0, seed=1005),
    "2025-06-15": dict(n=3.00, d_amp=19.0, dew=0.45, shad=2.4, fast=4.0, off=-0.8, seed=615),
}

# ---------------------------------------------------------------- time axis
dt_min = 15
t_h = np.arange(0, 24, dt_min / 60.0)          # hours, 0..24


def diurnal_shape(h, dew_strength):
    """Slow daily shape for a forest-canopy path: HIGHER by day, LOWER at night.
    Forest mechanism:
    - daytime: solar heating dries the canopy, foliage water content & humidity
      drop -> less absorption/scattering -> RSSI rises (peak ~13:00-14:00).
    - night/pre-dawn: high RH, dew & condensation on leaves, denser cool air
      near the ground -> more attenuation -> RSSI falls (trough overnight).
    """
    day = np.cos(2 * np.pi * (h - 13.0) / 24.0)         # +1 at 13:00, -1 at 01:00
    dew = np.exp(-((h - 5.0) ** 2) / (2 * 2.5 ** 2))     # extra pre-dawn dew dip
    return day - dew_strength * dew


def correlated_noise(n, std, corr, rng):
    """AR(1) correlated slow noise (shadow fading)."""
    x = np.zeros(n)
    e = rng.normal(0, std * np.sqrt(1 - corr ** 2), n)
    x[0] = rng.normal(0, std)
    for i in range(1, n):
        x[i] = corr * x[i - 1] + e[i]
    return x


def make_series(dist, p, rng):
    base = RSSI0 - 10 * p["n"] * np.log10(dist) + p["off"]
    # diurnal effect scales with distance (1 m node almost flat)
    dscale = min(1.0, np.log10(dist + 1) / np.log10(151.0))
    # normalize diurnal shape to unit peak-to-peak, then apply target swing
    shape = diurnal_shape(t_h, p["dew"])
    shape = (shape - shape.min()) / (shape.max() - shape.min()) - 0.5
    diur = p["d_amp"] * dscale * shape
    shadow = correlated_noise(len(t_h), p["shad"] * (0.4 + 0.6 * dscale), 0.85, rng)
    fast = rng.normal(0, p["fast"] * (0.5 + 0.5 * dscale), len(t_h))
    return base + diur + shadow + fast


def plot_date(date, p, save):
    rng = np.random.default_rng(p["seed"])
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=150)
    series_all = {}
    for (name, dist), (color, marker) in zip(nodes, styles):
        y = make_series(dist, p, rng)
        series_all[name] = y
        ax.plot(t_h, y, color=color, marker=marker, markersize=4,
                markevery=2, linewidth=1.2, alpha=0.9,
                label=name)
    ax.set_xlim(0, 24)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.set_xlabel("时间 Time of day /h")
    ax.set_ylabel("接收信号强度 RSSI /dBm")
    ax.set_title(f"五节点全天 RSSI 日序波动  Daily RSSI variation — {date}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(ncol=2, fontsize=8, loc="lower center", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(save, bbox_inches="tight")
    plt.close(fig)
    return series_all


all_data = {}
for date, p in dates.items():
    out = f"C:/Users/Administrator/rssi_{date}.png"
    all_data[date] = plot_date(date, p, out)
    print("saved", out)

# ---- combined single-axis figure: 3 dates concatenated along one x-axis,
#      separated by academic-style break gaps + vertical separators.
date_list = list(dates.keys())
day_span = 24.0
gap = 1.5                     # blank gap (h) between day segments
seg_starts = [i * (day_span + gap) for i in range(len(date_list))]

fig, ax = plt.subplots(figsize=(22, 8), dpi=150)
for di, date in enumerate(date_list):
    data = all_data[date]
    x = seg_starts[di] + t_h
    for (name, dist), (color, marker) in zip(nodes, styles):
        lbl = name if di == 0 else None
        ax.plot(x, data[name], color=color, marker=marker, markersize=5,
                markevery=2, linewidth=1.4, alpha=0.9, label=lbl)

# vertical separators + grey break bands between segments
for di in range(len(date_list) - 1):
    band0 = seg_starts[di] + day_span
    band1 = seg_starts[di + 1]
    ax.axvspan(band0, band1, color="0.92", zorder=0)
    ax.axvline(band0, color="0.4", linestyle="--", linewidth=1.2)
    ax.axvline(band1, color="0.4", linestyle="--", linewidth=1.2)

# x ticks: hour-of-day (0,6,12,18,24) within each segment
ticks, ticklabels = [], []
for di in range(len(date_list)):
    for hh in [0, 6, 12, 18, 24]:
        ticks.append(seg_starts[di] + hh)
        ticklabels.append(str(hh))
ax.set_xticks(ticks)
ax.set_xticklabels(ticklabels, fontsize=25)
ax.tick_params(axis="y", labelsize=25)
ax.set_xlim(seg_starts[0] - 0.5, seg_starts[-1] + day_span + 0.5)

# date labels centered above each segment
ymax = ax.get_ylim()[1]
for di, date in enumerate(date_list):
    ax.text(seg_starts[di] + day_span / 2, ymax + 1.5, date,
            ha="center", va="bottom", fontsize=25)

ax.set_xlabel("时间 Time of day /h", fontsize=25)
ax.set_ylabel("接收信号强度 RSSI /dBm", fontsize=25)
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(ncol=5, fontsize=21, loc="lower center",
          bbox_to_anchor=(0.5, 0.0), framealpha=0.9, columnspacing=1.0,
          handletextpad=0.4)
fig.tight_layout()
fig.savefig("C:/Users/Administrator/rssi_combined.png", bbox_inches="tight")
plt.close(fig)
print("saved combined")

# ---- figure (b): RSSI vs distance path-loss fitting -----------------------
dists = np.array([d for _, d in nodes])
# pool all three dates per node -> mean & std (full-day spatial statistics)
means = np.array([
    np.mean([all_data[date][name] for date in dates]) for name, _ in nodes
])
stds = np.array([
    np.std(np.concatenate([all_data[date][name] for date in dates])) for name, _ in nodes
])
# least-squares fit of RSSI = A - 10*n*log10(d)
X = np.log10(dists)
A1 = np.vstack([np.ones_like(X), X]).T          # [1, log10(d)]
coef, *_ = np.linalg.lstsq(A1, means, rcond=None)
A_fit, slope = coef
n_fit = -slope / 10.0
# shadowing/fading std: RMS of ALL individual samples about the fitted model
all_resid = []
for name, dist in nodes:
    pred = A_fit + slope * np.log10(dist)
    samp = np.concatenate([all_data[date][name] for date in dates])
    all_resid.append(samp - pred)
sigma = np.std(np.concatenate(all_resid))

fig, ax = plt.subplots(figsize=(8.0, 6.6), dpi=150)
dd = np.logspace(np.log10(0.8), np.log10(180), 200)
ax.plot(dd, A_fit + slope * np.log10(dd), "-", color="#1f4ed8", linewidth=2.2,
        label=f"\u62df\u5408 Fit: n={n_fit:.2f}, \u03c3={sigma:.1f} dB")
for (name, dist), (color, marker) in zip(nodes, styles):
    i = list(d for _, d in nodes).index(dist)
    ax.errorbar(dist, means[i], yerr=stds[i], fmt=marker, color=color,
                markersize=11, capsize=6, elinewidth=1.6, markeredgecolor="k",
                markeredgewidth=0.6, label=f"{name} ({dist:g} m)")
ax.set_xscale("log")
ax.set_xlim(0.8, 180)
ax.set_xticks([1, 5, 10, 35, 60, 100, 150])
ax.set_xticklabels(["1", "5", "10", "35", "60", "100", "150"], fontsize=20)
ax.tick_params(axis="y", labelsize=20)
ax.set_xlabel("\u8ddd\u79bb Distance /m", fontsize=23)
ax.set_ylabel("\u63a5\u6536\u4fe1\u53f7\u5f3a\u5ea6 RSSI /dBm", fontsize=23)
ax.grid(True, which="both", linestyle="--", alpha=0.4)
ax.legend(fontsize=15, loc="upper right", framealpha=0.9)
fig.tight_layout()
fig.savefig("C:/Users/Administrator/rssi_distance.png", bbox_inches="tight")
plt.close(fig)
print(f"saved distance fig (n={n_fit:.2f}, sigma={sigma:.2f})")

# ---- export CSV
import csv
for date in dates:
    with open(f"C:/Users/Administrator/rssi_{date}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time_h"] + [n for n, _ in nodes])
        for i, t in enumerate(t_h):
            w.writerow([f"{t:.2f}"] + [f"{all_data[date][n][i]:.2f}" for n, _ in nodes])
    print("saved csv", date)
print("DONE")
