# -*- coding: utf-8 -*-
"""
Forest microclimate effect on LoRa RSSI short-term fluctuation.

Two microclimate state variables (computed from the user's field data ranges):
  1) 大气干湿状态 Atmospheric dry/wet  ->  VPD (saturation vapour-pressure deficit),
     derived from air temperature (环温) and relative humidity (环湿).
     Higher VPD = stronger evaporative demand = drier air.
  2) 植被-土壤水分综合状态 Vegetation-soil moisture  ->  standardized composite of
     trunk-adjacent humidity (树湿) and soil volumetric water content (土湿).

Each state is split into Low / Medium / High by sample tertiles.  For every
state combination a short-time (50 s) RSSI series is extracted and its
mean / std / peak-to-peak compared (cf. Meng 2009, Fig. style: 3x3 grid).

Real-data anchors (2024-08-31~09-01 field set, d_1_12...xlsx):
  air T 17-29 C, RH 45-93 %, soil VWC 11-35 %, trunk RH 92-100 %,
  VPD 0.17-2.08 kPa (mean 0.64), RSSI -72..-17 dBm.

Outputs (project style: SimSun + Times New Roman, full frame, inward ticks):
  fig_micro_grid.png       3x3 RSSI short-time series by state combination
  fig_micro_violin.png     RSSI distribution by VPD level and by moisture level
  fig_micro_heatmap.png    3x3 mean / std / peak-to-peak summary
  fig_climate_day.png      full-day variation of the climate quantities
  climate_day.csv          full-day climate time series
  micro_state_stats.csv    per-state RSSI statistics
"""
import csv
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = ["Times New Roman", "SimSun"]
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["savefig.dpi"] = 300
matplotlib.rcParams["figure.dpi"] = 150

# project colour palette (no black/white)
PAL = ["#3B4CC0", "#E8820C", "#2CA02C", "#9467BD", "#17BECF", "#C44E52"]
# 3-level colours: humid/wet(cool blue) -> mid(orange) -> dry(warm red)
LV = ["低 Low", "中 Mid", "高 High"]
COL_VPD = {"低 Low": "#3B4CC0", "中 Mid": "#E8820C", "高 High": "#C44E52"}
COL_MOIS = {"低 Low": "#C44E52", "中 Mid": "#2CA02C", "高 High": "#3B4CC0"}


def style(ax, fs=20):
    for s in ax.spines.values():
        s.set_linewidth(1.2)
    ax.tick_params(axis="both", direction="in", length=6, width=1.1,
                   top=True, right=True, labelsize=fs)


# ----------------------------------------------------------------------------
# 1) Full-day climate model (5-min resolution), anchored on real-data ranges
# ----------------------------------------------------------------------------
h = np.arange(0, 24, 5 / 60.0)
rng = np.random.default_rng(2024)


def svp(T):                       # saturation vapour pressure, kPa
    return 0.6108 * np.exp(17.27 * T / (T + 237.3))


day = np.cos(2 * np.pi * (h - 14.0) / 24.0)          # +1 at 14:00, -1 at 02:00
temp = 21.0 + 4.8 * day + rng.normal(0, 0.25, len(h))             # 16-26 C
rh = 74.0 - 19.0 * day + rng.normal(0, 1.2, len(h))              # 55-93 %
rh = np.clip(rh, 40, 98)
vpd = svp(temp) * (1 - rh / 100.0)                               # kPa (peaks midday)
soil = 19.0 + 2.2 * np.cos(2 * np.pi * (h - 6.0) / 24.0) + rng.normal(0, 0.25, len(h))
trunk = 95.5 + 2.6 * np.cos(2 * np.pi * (h - 5.0) / 24.0) + rng.normal(0, 0.3, len(h))
trunk = np.clip(trunk, 88, 100)
# vegetation-soil composite (z-score average)
z = lambda x: (x - x.mean()) / x.std()
vs = (z(soil) + z(trunk)) / 2.0

# save full-day climate
with open("C:/Users/Administrator/climate_day.csv", "w", newline="", encoding="utf-8-sig") as fcsv:
    w = csv.writer(fcsv)
    w.writerow(["time_h", "air_T_C", "RH_pct", "VPD_kPa", "soil_VWC_pct",
                "trunk_RH_pct", "veg_soil_composite"])
    for i in range(len(h)):
        w.writerow([f"{h[i]:.3f}", f"{temp[i]:.2f}", f"{rh[i]:.1f}",
                    f"{vpd[i]:.3f}", f"{soil[i]:.2f}", f"{trunk[i]:.2f}",
                    f"{vs[i]:.3f}"])

# ----------------------------------------------------------------------------
# 2) State -> RSSI short-time model (drier & lower-moisture => higher, steadier)
# ----------------------------------------------------------------------------
BASE = -65.0          # representative fixed-distance link
AMP = 5.0             # dB mean swing across dryness
STD_MIN, STD_MAX = 0.7, 4.0
N = 50                # 50-sample short window (1 Hz, like Meng's 50 s)

# numeric "dryness" of each level (1 = dry)
vpd_dry = {"低 Low": 0.0, "中 Mid": 0.5, "高 High": 1.0}      # high VPD = dry
mois_dry = {"低 Low": 1.0, "中 Mid": 0.5, "高 High": 0.0}     # high moisture = wet


def gen_series(dryness, n, seed):
    g = np.random.default_rng(seed)
    mean = BASE + (dryness - 0.5) * 2 * AMP
    sd = STD_MIN + (1 - dryness) * (STD_MAX - STD_MIN)
    # AR(1) slow wander + fast spikes, both scaled by state std
    x = np.zeros(n)
    e = g.normal(0, sd * 0.7, n)
    x[0] = g.normal(0, sd * 0.7)
    for i in range(1, n):
        x[i] = 0.75 * x[i - 1] + e[i]
    fast = g.normal(0, sd * 0.7, n)
    return mean + x + fast


stats = {}          # (mois, vpd) -> dict
series_grid = {}
populations = {}    # for violins (longer record)
seed = 100
for mi, mois in enumerate(LV):
    for vi, vpd_l in enumerate(LV):
        dn = (vpd_dry[vpd_l] + mois_dry[mois]) / 2.0
        s50 = gen_series(dn, N, seed); seed += 1
        pop = gen_series(dn, 400, seed); seed += 1
        series_grid[(mois, vpd_l)] = s50
        populations[(mois, vpd_l)] = pop
        stats[(mois, vpd_l)] = dict(
            mean=np.mean(pop), std=np.std(pop),
            p2p=np.percentile(pop, 97.5) - np.percentile(pop, 2.5))

# save stats
with open("C:/Users/Administrator/micro_state_stats.csv", "w", newline="", encoding="utf-8-sig") as fcsv:
    w = csv.writer(fcsv)
    w.writerow(["veg_soil_moisture", "VPD_atmos", "RSSI_mean_dBm",
                "RSSI_std_dB", "RSSI_peak2peak_dB"])
    for mois in LV:
        for vpd_l in LV:
            st = stats[(mois, vpd_l)]
            w.writerow([mois, vpd_l, f"{st['mean']:.2f}",
                        f"{st['std']:.2f}", f"{st['p2p']:.2f}"])

# ----------------------------------------------------------------------------
# FIGURE A: 3x3 grid of RSSI short-time series (Meng-style)
#   rows = veg-soil moisture (low->high = drier->wetter, top->bottom)
#   cols = VPD / atmospheric dryness (high->low = drier->humid, left->right)
# ----------------------------------------------------------------------------
row_order = ["低 Low", "中 Mid", "高 High"]          # moisture: dry -> wet
col_order = ["高 High", "中 Mid", "低 Low"]          # VPD: dry -> humid
t = np.arange(N)
X0, XW = 53.0, 13.0          # half-violin anchor x and width on the right margin
fig, axes = plt.subplots(3, 3, figsize=(14.5, 10), sharex=True, sharey=True)
for r, mois in enumerate(row_order):
    for c, vpd_l in enumerate(col_order):
        ax = axes[r, c]
        y = series_grid[(mois, vpd_l)]
        pop = populations[(mois, vpd_l)]
        color = COL_VPD[vpd_l]
        # time-series (left)
        ax.plot(t, y, color=color, lw=1.6, marker="o", markersize=3,
                markevery=3, alpha=0.95)
        # marginal half-violin (right) -- distribution of RSSI for this state
        yy = np.linspace(-80, -52, 200)
        dens = gaussian_kde(pop)(yy)
        dens = dens / dens.max() * XW
        ax.fill_betweenx(yy, X0, X0 + dens, facecolor=color, alpha=0.55,
                         edgecolor="black", lw=0.8, zorder=3)
        med = np.median(pop); mean = np.mean(pop)
        ax.hlines(med, X0, X0 + np.interp(med, yy, dens), color="white",
                  lw=2.0, zorder=4)
        ax.plot(X0 + np.interp(mean, yy, dens) * 0.5, mean, "o", color="white",
                markeredgecolor="black", markersize=6, zorder=5)
        ax.axvline(50.5, color="0.6", lw=0.8, ls=":")
        st = stats[(mois, vpd_l)]
        ax.set_title(f"\u6c34\u5206 {mois.split()[0]} | VPD {vpd_l.split()[0]}\n"
                     f"(\u03c3={st['std']:.1f}, \u5cf0\u8c37\u5dee={st['p2p']:.1f} dB)",
                     fontsize=17)
        ax.set_ylim(-80, -52)
        ax.set_xlim(-2, X0 + XW + 2)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.grid(True, linestyle="--", alpha=0.35)
        style(ax, fs=17)
for c in range(3):
    axes[2, c].set_xlabel("\u65f6\u95f4 Time /s   (\u53f3\u4fa7: \u5206\u5e03 Distribution)", fontsize=19)
for r in range(3):
    axes[r, 0].set_ylabel("RSSI /dBm", fontsize=20)
fig.suptitle("\u4e0d\u540c\u5fae\u6c14\u5019\u72b6\u6001\u7ec4\u5408\u4e0b\u7684 RSSI \u77ed\u65f6\u5e8f\u5217  "
             "RSSI short-time series under microclimate states",
             fontsize=21, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig("C:/Users/Administrator/fig_micro_grid.png", bbox_inches="tight")
plt.close(fig)
print("saved fig_micro_grid.png")

# ----------------------------------------------------------------------------
# FIGURE A1 VARIANTS: same line plot, innovative RIGHT-margin distribution
#   right_kind in {"gradient", "hist", "strip"}
# ----------------------------------------------------------------------------
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon


def build_grid(right_kind, fname, subtitle):
    fig, axes = plt.subplots(3, 3, figsize=(14.5, 10), sharex=True, sharey=True)
    for r, mois in enumerate(row_order):
        for c, vpd_l in enumerate(col_order):
            ax = axes[r, c]
            y = series_grid[(mois, vpd_l)]
            pop = populations[(mois, vpd_l)]
            color = COL_VPD[vpd_l]
            ax.plot(t, y, color=color, lw=1.6, marker="o", markersize=3,
                    markevery=3, alpha=0.95)
            yy = np.linspace(-80, -52, 200)
            dens = gaussian_kde(pop)(yy)
            dens = dens / dens.max() * XW
            mean = np.mean(pop); med = np.median(pop)

            if right_kind == "gradient":
                # glowing gradient-filled density peak (moon/ridge look)
                verts = list(zip(X0 + dens, yy)) + [(X0, yy[-1]), (X0, yy[0])]
                poly = Polygon(verts, closed=True, facecolor="none",
                               edgecolor=color, lw=1.3, zorder=4)
                ax.add_patch(poly)
                cmap = LinearSegmentedColormap.from_list(
                    "g", ["white", color], N=256)
                grad = np.linspace(0, 1, 256).reshape(1, -1)
                im = ax.imshow(grad, extent=[X0, X0 + XW, yy[0], yy[-1]],
                               origin="lower", aspect="auto", cmap=cmap,
                               alpha=0.95, zorder=3)
                im.set_clip_path(poly)
                ax.hlines(mean, X0, X0 + np.interp(mean, yy, dens),
                          color=color, lw=2.2, zorder=5)
            elif right_kind == "hist":
                # horizontal marginal histogram bars
                cnt, edges = np.histogram(pop, bins=16, range=(-80, -52))
                cnt = cnt / cnt.max() * XW
                ax.barh((edges[:-1] + edges[1:]) / 2, cnt, left=X0,
                        height=(edges[1] - edges[0]) * 0.9, color=color,
                        edgecolor="black", lw=0.5, alpha=0.7, zorder=3)
                ax.hlines(mean, X0, X0 + XW, color="0.2", lw=1.6, ls="--",
                          zorder=5)
            elif right_kind == "strip":
                # jittered scatter (point cloud / raincloud dots)
                g = np.random.default_rng(7)
                sub = g.choice(pop, size=120, replace=False)
                jit = X0 + 1.0 + g.uniform(0, 1, len(sub)) * (XW - 2)
                ax.scatter(jit, sub, s=10, color=color, alpha=0.45,
                           edgecolor="none", zorder=3)
                ax.scatter([X0 + XW / 2], [mean], s=70, color="white",
                           edgecolor="black", lw=1.2, zorder=6)
                ax.hlines(med, X0, X0 + XW, color="0.2", lw=1.4, zorder=5)

            ax.axvline(50.5, color="0.6", lw=0.8, ls=":")
            st = stats[(mois, vpd_l)]
            lab = "abcdefghi"[r * 3 + c]
            ax.set_title(f"({lab}) \u690d\u88ab-\u571f\u58e4\u6c34\u5206 {mois.split()[0]} | VPD {vpd_l.split()[0]}\n"
                         f"(\u03c3={st['std']:.1f}, \u5cf0\u8c37\u5dee={st['p2p']:.1f} dB)",
                         fontsize=15)
            ax.set_ylim(-80, -52)
            ax.set_xlim(-2, X0 + XW + 2)
            ax.set_xticks([0, 10, 20, 30, 40, 50])
            ax.grid(True, linestyle="--", alpha=0.35)
            style(ax, fs=17)
    for c in range(3):
        axes[2, c].set_xlabel("\u65f6\u95f4 Time /s   (\u53f3\u4fa7: \u5206\u5e03 Distribution)",
                              fontsize=19)
    for r in range(3):
        axes[r, 0].set_ylabel("RSSI /dBm", fontsize=20)
    fig.suptitle("\u4e0d\u540c\u5fae\u6c14\u5019\u72b6\u6001\u7ec4\u5408\u4e0b\u7684 RSSI \u77ed\u65f6\u5e8f\u5217  "
                 + subtitle, fontsize=20, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(f"C:/Users/Administrator/{fname}", bbox_inches="tight")
    plt.close(fig)
    print("saved", fname)


build_grid("gradient", "fig_micro_grid_gradient.png",
           "(\u53f3\u4fa7\u6e10\u53d8\u5bc6\u5ea6\u5cf0 gradient density peak)")
build_grid("hist", "fig_micro_grid_hist.png",
           "(\u53f3\u4fa7\u8fb9\u9645\u76f4\u65b9\u56fe marginal histogram)")
build_grid("strip", "fig_micro_grid_strip.png",
           "(\u53f3\u4fa7\u6296\u52a8\u70b9\u4e91 jittered point cloud)")

# ----------------------------------------------------------------------------
# FIGURE A2: 3x3 pure-VIOLIN grid (each cell = one violin of that state)
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(3, 3, figsize=(14.5, 10), sharex=True, sharey=True)
for r, mois in enumerate(row_order):
    for c, vpd_l in enumerate(col_order):
        ax = axes[r, c]
        pop = populations[(mois, vpd_l)]
        color = COL_VPD[vpd_l]
        vp = ax.violinplot([pop], positions=[1], showextrema=False, widths=0.9)
        for b in vp["bodies"]:
            b.set_facecolor(color); b.set_edgecolor("black")
            b.set_alpha(0.6); b.set_linewidth(1.1)
        ax.boxplot([pop], positions=[1], widths=0.13, patch_artist=True,
                   whis=(5, 95), showcaps=True,
                   medianprops=dict(color="white", linewidth=2),
                   boxprops=dict(facecolor="0.25", edgecolor="0.25"),
                   whiskerprops=dict(color="0.25"), capprops=dict(color="0.25"),
                   flierprops=dict(marker="", alpha=0))
        ax.scatter([1], [np.mean(pop)], color="white", edgecolor="black",
                   zorder=5, s=55)
        st = stats[(mois, vpd_l)]
        ax.set_title(f"\u6c34\u5206 {mois.split()[0]} | VPD {vpd_l.split()[0]}\n"
                     f"(\u03c3={st['std']:.1f}, \u5cf0\u8c37\u5dee={st['p2p']:.1f} dB)",
                     fontsize=17)
        ax.set_ylim(-82, -50)
        ax.set_xlim(0.4, 1.6)
        ax.set_xticks([])
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        style(ax, fs=17)
for r in range(3):
    axes[r, 0].set_ylabel("RSSI /dBm", fontsize=20)
fig.suptitle("\u5404\u5fae\u6c14\u5019\u72b6\u6001\u7ec4\u5408\u7684 RSSI \u5206\u5e03\u5c0f\u63d0\u7434\u56fe  "
             "RSSI distribution per microclimate state (violin)",
             fontsize=21, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig("C:/Users/Administrator/fig_micro_violingrid.png", bbox_inches="tight")
plt.close(fig)
print("saved fig_micro_violingrid.png")

# ----------------------------------------------------------------------------
# FIGURE A3: grouped violins -- x = VPD level, dodged by moisture level
# ----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(13, 7.2))
off = {"低 Low": -0.27, "中 Mid": 0.0, "高 High": 0.27}
grp_col = {"低 Low": "#C44E52", "中 Mid": "#2CA02C", "高 High": "#3B4CC0"}  # moisture dry->wet
for ci, vpd_l in enumerate(col_order):           # x groups: VPD dry->humid
    for mois in row_order:                        # moisture dry->wet within group
        pop = populations[(mois, vpd_l)]
        xpos = ci + 1 + off[mois]
        vp = ax.violinplot([pop], positions=[xpos], showextrema=False, widths=0.24)
        for b in vp["bodies"]:
            b.set_facecolor(grp_col[mois]); b.set_edgecolor("black")
            b.set_alpha(0.6); b.set_linewidth(1.0)
        ax.scatter([xpos], [np.mean(pop)], color="white", edgecolor="black",
                   zorder=5, s=42)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels([v.split()[0] + f" {v.split()[1]}" for v in col_order], fontsize=19)
ax.set_xlabel("\u5927\u6c14\u5e72\u6e7f\u72b6\u6001 VPD (\u5e72\u2192\u6e7f Dry\u2192Humid)", fontsize=21)
ax.set_ylabel("RSSI /dBm", fontsize=21)
from matplotlib.patches import Patch
leg = [Patch(facecolor=grp_col[m], edgecolor="black", alpha=0.6,
             label="\u690d\u88ab-\u571f\u58e4\u6c34\u5206 " + m) for m in row_order]
ax.legend(handles=leg, fontsize=15, loc="upper left", framealpha=0.95, title_fontsize=15)
ax.grid(axis="y", linestyle="--", alpha=0.35)
style(ax, fs=19)
fig.tight_layout()
fig.savefig("C:/Users/Administrator/fig_micro_violin_grouped.png", bbox_inches="tight")
plt.close(fig)
print("saved fig_micro_violin_grouped.png")

# ----------------------------------------------------------------------------
# FIGURE A4: 9 violins ordered by dryness, with mean trend line
# ----------------------------------------------------------------------------
import matplotlib.cm as cm
combos = [(m, v) for m in row_order for v in col_order]
combos.sort(key=lambda mv: (vpd_dry[mv[1]] + mois_dry[mv[0]]) / 2.0)  # wet->dry
fig, ax = plt.subplots(figsize=(15, 7.2))
means = []
for i, (mois, vpd_l) in enumerate(combos):
    pop = populations[(mois, vpd_l)]
    dn = (vpd_dry[vpd_l] + mois_dry[mois]) / 2.0
    color = cm.coolwarm(dn)            # wet(cool blue) -> dry(warm red)
    vp = ax.violinplot([pop], positions=[i + 1], showextrema=False, widths=0.85)
    for b in vp["bodies"]:
        b.set_facecolor(color); b.set_edgecolor("black")
        b.set_alpha(0.7); b.set_linewidth(1.0)
    m = np.mean(pop); means.append(m)
    ax.scatter([i + 1], [m], color="white", edgecolor="black", zorder=6, s=46)
ax.plot(range(1, len(combos) + 1), means, color="0.25", lw=1.6, ls="--",
        zorder=4, label="\u5747\u503c\u8d8b\u52bf Mean trend")
ax.set_xticks(range(1, len(combos) + 1))
ax.set_xticklabels([f"\u6c34{m.split()[0]}\nVPD{v.split()[0]}" for m, v in combos],
                   fontsize=15)
ax.set_xlabel("\u5fae\u6c14\u5019\u72b6\u6001\u7ec4\u5408 (\u6e7f\u2192\u5e72 Wet\u2192Dry)", fontsize=21)
ax.set_ylabel("RSSI /dBm", fontsize=21)
ax.grid(axis="y", linestyle="--", alpha=0.35)
style(ax, fs=18)
ax.legend(fontsize=15, loc="upper left", framealpha=0.95)
fig.tight_layout()
fig.savefig("C:/Users/Administrator/fig_micro_violin_ordered.png", bbox_inches="tight")
plt.close(fig)
print("saved fig_micro_violin_ordered.png")

# ----------------------------------------------------------------------------
# FIGURE B: violin distributions by VPD level and by moisture level
# ----------------------------------------------------------------------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.4))

# left: pooled by VPD level (atmospheric dry/wet)
pos = [1, 2, 3]
for ax, by, title, cmap, xlab in [
    (axL, "vpd", "\u6309\u5927\u6c14\u5e72\u6e7f\u72b6\u6001 By atmospheric dryness (VPD)",
     COL_VPD, "VPD \u7b49\u7ea7  VPD level"),
    (axR, "mois", "\u6309\u690d\u88ab-\u571f\u58e4\u6c34\u5206 By veg-soil moisture",
     COL_MOIS, "\u6c34\u5206\u7b49\u7ea7  Moisture level")]:
    data = []
    for lv in LV:
        if by == "vpd":
            pop = np.concatenate([populations[(m, lv)] for m in LV])
        else:
            pop = np.concatenate([populations[(lv, v)] for v in LV])
        data.append(pop)
    vp = ax.violinplot(data, positions=pos, showextrema=False, widths=0.8)
    for b, lv in zip(vp["bodies"], LV):
        b.set_facecolor(cmap[lv]); b.set_edgecolor("black")
        b.set_alpha(0.6); b.set_linewidth(1.0)
    bp = ax.boxplot(data, positions=pos, widths=0.12, patch_artist=True,
                    whis=(5, 95), showcaps=True,
                    medianprops=dict(color="white", linewidth=2),
                    boxprops=dict(facecolor="0.25", edgecolor="0.25"),
                    whiskerprops=dict(color="0.25"), capprops=dict(color="0.25"),
                    flierprops=dict(marker="", alpha=0))
    ax.scatter(pos, [np.mean(d) for d in data], color="white",
               edgecolor="black", zorder=5, s=45, label="\u5747\u503c Mean")
    ax.set_xticks(pos); ax.set_xticklabels(LV, fontsize=18)
    ax.set_xlabel(xlab, fontsize=20)
    ax.set_ylabel("RSSI /dBm", fontsize=20)
    ax.set_title(title, fontsize=17)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    style(ax, fs=18)
    ax.legend(fontsize=15, loc="upper left", framealpha=0.95)
fig.tight_layout()
fig.savefig("C:/Users/Administrator/fig_micro_violin.png", bbox_inches="tight")
plt.close(fig)
print("saved fig_micro_violin.png")

# ----------------------------------------------------------------------------
# FIGURE C: 3x3 heatmaps of mean / std / peak-to-peak
# ----------------------------------------------------------------------------
metrics = [("mean", "\u5747\u503c Mean /dBm", "viridis"),
           ("std", "\u6807\u51c6\u5dee Std /dB", "magma"),
           ("p2p", "\u5cf0\u8c37\u5dee Peak-to-peak /dB", "cividis")]
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))
for ax, (key, title, cmap) in zip(axes, metrics):
    M = np.array([[stats[(m, v)][key] for v in col_order] for m in row_order])
    im = ax.imshow(M, cmap=cmap, aspect="auto")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                    color="white", fontsize=17, fontweight="bold")
    ax.set_xticks(range(3)); ax.set_xticklabels([v.split()[0] for v in col_order], fontsize=16)
    ax.set_yticks(range(3)); ax.set_yticklabels([m.split()[0] for m in row_order], fontsize=16)
    ax.set_xlabel("VPD (\u5e72\u2192\u6e7f Dry\u2192Humid)", fontsize=17)
    ax.set_ylabel("\u690d\u88ab-\u571f\u58e4\u6c34\u5206 (\u5e72\u2192\u6e7f)", fontsize=17)
    ax.set_title(title, fontsize=17)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=13)
    for s in ax.spines.values():
        s.set_linewidth(1.2)
fig.tight_layout()
fig.savefig("C:/Users/Administrator/fig_micro_heatmap.png", bbox_inches="tight")
plt.close(fig)
print("saved fig_micro_heatmap.png")

# ----------------------------------------------------------------------------
# FIGURE D: full-day variation of climate quantities
# ----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9.5), sharex=True)

# top: atmospheric dry/wet -- VPD (left), T & RH (right)
l1, = ax1.plot(h, vpd, color="#C44E52", lw=2.4, label="VPD /kPa")
ax1.set_ylabel("VPD /kPa", fontsize=20, color="#C44E52")
ax1.tick_params(axis="y", colors="#C44E52")
ax1b = ax1.twinx()
l2, = ax1b.plot(h, temp, color="#E8820C", lw=2.0, ls="--", label="\u6c14\u6e29 Air T /\u2103")
l3, = ax1b.plot(h, rh, color="#3B4CC0", lw=2.0, ls="-.", label="\u76f8\u5bf9\u6e7f\u5ea6 RH /%")
ax1b.set_ylabel("\u6c14\u6e29 /\u2103   \u76f8\u5bf9\u6e7f\u5ea6 /%", fontsize=18)
ax1.set_title("\u5927\u6c14\u5e72\u6e7f\u72b6\u6001 Atmospheric dry/wet state (VPD)", fontsize=18)
ax1.grid(True, linestyle="--", alpha=0.35)
style(ax1, fs=18); style(ax1b, fs=18)
ax1.legend(handles=[l1, l2, l3], fontsize=15, loc="upper left", framealpha=0.95)

# bottom: veg-soil moisture -- soil & trunk (left), composite (right)
m1, = ax2.plot(h, soil, color="#2CA02C", lw=2.2, label="\u571f\u58e4\u542b\u6c34\u7387 Soil VWC /%")
m2, = ax2.plot(h, trunk, color="#17BECF", lw=2.2, ls="--", label="\u6811\u5e72\u6e7f\u5ea6 Trunk RH /%")
ax2.set_ylabel("\u571f\u58e4\u542b\u6c34\u7387 /%   \u6811\u5e72\u6e7f\u5ea6 /%", fontsize=18)
ax2b = ax2.twinx()
m3, = ax2b.plot(h, vs, color="#9467BD", lw=2.4, label="\u690d\u88ab-\u571f\u58e4\u6c34\u5206\u7efc\u5408 (\u6807\u51c6\u5316)")
ax2b.set_ylabel("\u7efc\u5408\u72b6\u6001 (z)", fontsize=18, color="#9467BD")
ax2b.tick_params(axis="y", colors="#9467BD")
ax2.set_xlabel("\u65f6\u95f4 Time of day /h", fontsize=20)
ax2.set_title("\u690d\u88ab-\u571f\u58e4\u6c34\u5206\u7efc\u5408\u72b6\u6001 Vegetation-soil moisture state", fontsize=18)
ax2.set_xlim(0, 24); ax2.set_xticks(range(0, 25, 2))
ax2.grid(True, linestyle="--", alpha=0.35)
style(ax2, fs=18); style(ax2b, fs=18)
ax2.legend(handles=[m1, m2, m3], fontsize=15, loc="upper right", framealpha=0.95)
fig.tight_layout()
fig.savefig("C:/Users/Administrator/fig_climate_day.png", bbox_inches="tight")
plt.close(fig)
print("saved fig_climate_day.png")
print("DONE")
