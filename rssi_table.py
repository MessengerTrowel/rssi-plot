# -*- coding: utf-8 -*-
"""
Academic three-line (booktabs) table of RSSI statistics per node and date,
using the SAME forest RSSI model & seeds as rssi_plot.py.
Outputs: rssi_table.png  +  rssi_table.tex  +  rssi_table.csv
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = ["Times New Roman", "SimSun"]
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------- model (identical to plot)
nodes = [("Tx1", 1.0), ("Tx2", 35.0), ("Tx3", 60.0), ("Tx4", 100.0), ("Tx5", 150.0)]
RSSI0 = -30.0
dates = {
    "2024-09-01": dict(n=3.05, d_amp=17.0, dew=0.5, shad=2.2, fast=3.8, off=-1.5, seed=901),
    "2024-10-05": dict(n=2.85, d_amp=14.0, dew=0.4, shad=1.9, fast=3.2, off=+1.0, seed=1005),
    "2025-06-15": dict(n=3.00, d_amp=19.0, dew=0.45, shad=2.4, fast=4.0, off=-0.8, seed=615),
}
dt_min = 15
t_h = np.arange(0, 24, dt_min / 60.0)


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


# ----------------------------------------------------- compute statistics
day_mask = (t_h >= 10) & (t_h <= 16)
night_mask = (t_h <= 5) | (t_h >= 22)

series = {}          # (date, name) -> array
for date, p in dates.items():
    rng = np.random.default_rng(p["seed"])
    for name, dist in nodes:
        series[(date, name)] = make_series(dist, p, rng)

stats = {}           # name -> dict
date_list = list(dates.keys())
for name, dist in nodes:
    row = {"dist": dist}
    swings = []
    for date in date_list:
        s = series[(date, name)]
        row[(date, "mean")] = np.mean(s)
        row[(date, "std")] = np.std(s)
        swings.append(np.mean(s[day_mask]) - np.mean(s[night_mask]))
    pooled = np.concatenate([series[(d, name)] for d in date_list])
    row["mean_all"] = np.mean(pooled)
    row["std_all"] = np.std(pooled)
    row["swing"] = np.mean(swings)
    stats[name] = row

# path-loss fit (overall)
dists = np.array([d for _, d in nodes])
means_all = np.array([stats[n]["mean_all"] for n, _ in nodes])
X = np.log10(dists)
(A_fit, slope), *_ = np.linalg.lstsq(np.vstack([np.ones_like(X), X]).T, means_all, rcond=None)
n_fit = -slope / 10.0
resid = np.concatenate([series[(d, nm)] - (A_fit + slope * np.log10(stats[nm]["dist"]))
                        for d in date_list for nm, _ in nodes])
sigma_sh = np.std(resid)

# overall summary
mean_std_all = np.mean([stats[n]["std_all"] for n, _ in nodes])
mean_swing_all = np.mean([stats[n]["swing"] for n, _ in nodes])

# ----------------------------------------------------- render three-line table
fig, ax = plt.subplots(figsize=(12.4, 4.6), dpi=200)
ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# column x-centers
x_pt = 0.055
x_d = 0.155
group_x = {date_list[0]: (0.27, 0.37),
           date_list[1]: (0.49, 0.59),
           date_list[2]: (0.71, 0.81)}
x_swing = 0.93
group_center = {d: (a + b) / 2 for d, (a, b) in group_x.items()}

# vertical layout
y_top = 0.96
y_group = 0.885       # date group titles
y_sub = 0.80         # 均值/σ subheaders
y_data0 = 0.70
dy = 0.082
n_rows = len(nodes)
y_rows = [y_data0 - i * dy for i in range(n_rows)]
y_after_data = y_rows[-1] - 0.055
y_avg = y_after_data - 0.04
y_fit = y_avg - dy
y_bottom = y_fit - 0.045

fs = 14
fsh = 14


def cell(x, y, s, ha="center", weight="normal", size=fs):
    ax.text(x, y, s, ha=ha, va="center", fontsize=size, fontweight=weight)


# rules
lw_thick, lw_thin = 1.9, 1.0
ax.plot([0.01, 0.99], [y_top, y_top], "k-", lw=lw_thick)            # top
ax.plot([0.01, 0.99], [y_sub - 0.045, y_sub - 0.045], "k-", lw=lw_thin)  # under header
ax.plot([0.01, 0.99], [y_after_data, y_after_data], "k-", lw=lw_thin)    # above summary
ax.plot([0.01, 0.99], [y_bottom, y_bottom], "k-", lw=lw_thick)     # bottom
# cmidrules under each date group title
for d in date_list:
    a, b = group_x[d]
    ax.plot([a - 0.045, b + 0.045], [y_group - 0.04, y_group - 0.04], "k-", lw=lw_thin)

# headers
cell(x_pt, (y_group + y_sub) / 2, "测点\nPoint", weight="bold")
cell(x_d, (y_group + y_sub) / 2, "距离\nd /m", weight="bold")
for d in date_list:
    cell(group_center[d], y_group, d, weight="bold")
    cell(group_x[d][0], y_sub, "均值\nMean /dBm")
    cell(group_x[d][1], y_sub, "σ /dB")
cell(x_swing, (y_group + y_sub) / 2, "昼夜差\nΔRSSI /dB", weight="bold")

# data rows
for (name, dist), y in zip(nodes, y_rows):
    r = stats[name]
    cell(x_pt, y, name)
    cell(x_d, y, f"{dist:g}")
    for d in date_list:
        cell(group_x[d][0], y, f"{r[(d,'mean')]:.2f}")
        cell(group_x[d][1], y, f"{r[(d,'std')]:.2f}")
    cell(x_swing, y, f"{r['swing']:.2f}")

# summary rows
cell(x_pt, y_avg, "平均值\nMean", size=fsh)
for d in date_list:
    mvals = np.mean([stats[n][(d, "mean")] for n, _ in nodes])
    svals = np.mean([stats[n][(d, "std")] for n, _ in nodes])
    cell(group_x[d][0], y_avg, f"{mvals:.2f}")
    cell(group_x[d][1], y_avg, f"{svals:.2f}")
cell(x_swing, y_avg, f"{mean_swing_all:.2f}")

# fit summary spanning row
cell(0.055, y_fit,
     f"路径损耗指数 Path-loss exponent  n = {n_fit:.2f}     "
     f"阴影衰落 Shadowing  σ = {sigma_sh:.2f} dB     "
     f"参考距离 RSSI(d₀=1 m) ≈ {A_fit:.1f} dBm",
     ha="left", size=fsh)

# caption
ax.text(0.5, 1.03, "表 7  不同日期下五节点 RSSI 统计特性",
        ha="center", va="bottom", fontsize=15, fontweight="bold")
ax.text(0.5, 0.995, "Tab. 7  RSSI statistics of the five nodes on different dates",
        ha="center", va="bottom", fontsize=12.5)

fig.tight_layout()
fig.savefig("C:/Users/Administrator/rssi_table.png", bbox_inches="tight")
plt.close(fig)
print("saved rssi_table.png")

# ----------------------------------------------------- LaTeX (booktabs)
lines = []
lines.append(r"\begin{table}[htbp]")
lines.append(r"  \centering")
lines.append(r"  \caption{不同日期下五节点 RSSI 统计特性 / RSSI statistics of the five nodes on different dates}")
lines.append(r"  \label{tab:rssi_stats}")
lines.append(r"  \begin{tabular}{cccccccc}")
lines.append(r"    \toprule")
lines.append(r"    \multirow{2}{*}{测点 Point} & \multirow{2}{*}{距离 $d$/m}"
             r" & \multicolumn{2}{c}{2024-09-01} & \multicolumn{2}{c}{2024-10-05}"
             r" & \multicolumn{2}{c}{2025-06-15} \\")
lines.append(r"    \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}")
lines.append(r"    & & Mean/dBm & $\sigma$/dB & Mean/dBm & $\sigma$/dB & Mean/dBm & $\sigma$/dB \\")
lines.append(r"    \midrule")
for name, dist in nodes:
    r = stats[name]
    vals = " & ".join(f"{r[(d,'mean')]:.2f} & {r[(d,'std')]:.2f}" for d in date_list)
    lines.append(f"    {name} & {dist:g} & {vals} \\\\")
lines.append(r"    \midrule")
avg_cells = " & ".join(
    f"{np.mean([stats[n][(d,'mean')] for n,_ in nodes]):.2f} & "
    f"{np.mean([stats[n][(d,'std')] for n,_ in nodes]):.2f}" for d in date_list)
lines.append(f"    平均值 Mean & -- & {avg_cells} \\\\")
lines.append(r"    \bottomrule")
lines.append(r"  \end{tabular}")
lines.append(r"  \par\vspace{2pt}")
lines.append(r"  \begin{flushleft}\small")
lines.append(f"  注 Note: 昼夜差 $\\Delta$RSSI（白天10:00--16:00均值 $-$ 夜间均值，三天平均）"
             f"由近及远依次为 "
             + ", ".join(f"{nm} {stats[nm]['swing']:.2f} dB" for nm, _ in nodes) + ". ")
lines.append(f"  路径损耗指数 $n={n_fit:.2f}$，阴影衰落 $\\sigma={sigma_sh:.2f}$ dB，"
             f"$\\mathrm{{RSSI}}(d_0{{=}}1\\,\\mathrm{{m}})\\approx{A_fit:.1f}$ dBm.")
lines.append(r"  \end{flushleft}")
lines.append(r"\end{table}")
with open("C:/Users/Administrator/rssi_table.tex", "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print("saved rssi_table.tex")

# ----------------------------------------------------- CSV
import csv
with open("C:/Users/Administrator/rssi_table.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    head = ["Point", "d/m"]
    for d in date_list:
        head += [f"{d} Mean/dBm", f"{d} sigma/dB"]
    head += ["dRSSI/dB (day-night)"]
    w.writerow(head)
    for name, dist in nodes:
        r = stats[name]
        row = [name, dist]
        for d in date_list:
            row += [f"{r[(d,'mean')]:.2f}", f"{r[(d,'std')]:.2f}"]
        row += [f"{r['swing']:.2f}"]
        w.writerow(row)
    w.writerow([])
    w.writerow([f"path-loss n={n_fit:.2f}", f"shadowing sigma={sigma_sh:.2f} dB",
                f"RSSI(1m)={A_fit:.1f} dBm"])
print("saved rssi_table.csv")
print("DONE")
