# -*- coding: utf-8 -*-
"""
Redraw Fig.3: measured RSSI vs communication distance under four scenarios,
from the user's raw data (fig3_rssi_1m_scenarios.csv), in the project style:
Chinese SimSun + English Times New Roman, font size 25, axis titles with /unit.
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

matplotlib.rcParams["font.family"] = ["Times New Roman", "SimSun"]
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["font.size"] = 25

CSV = ("C:/Users/Administrator/attachments/55215357-30f0-4477-b6f3-a394b47342fc/"
       "cUsersAdministratorDocumentsCodex2026-05-26files-mentioned-by-the-user-"
       "433datafig3_rssi_1m_scenarios.csv")

# scenario order + bilingual label + style (open markers, like the reference)
scenarios = [
    ("\u5bc6\u6797",     "\u5bc6\u6797 Dense forest",  "#000000", "D"),
    ("\u758f\u5bc6\u7ed3\u5408", "\u758f\u5bc6\u7ed3\u5408 Mixed",  "#E8820C", "o"),
    ("\u758f\u6797",     "\u758f\u6797 Sparse forest", "#1f4ed8", "^"),
    ("\u7a7a\u65f7\u8349\u5730", "\u7a7a\u65f7\u8349\u5730 Open grassland", "#e8000b", "s"),
]

# read data
data = {key: ([], []) for key, *_ in scenarios}
with open(CSV, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        sc = row["scenario"].strip()
        val = row["rssi_dbm"].strip()
        if sc in data and val != "":
            data[sc][0].append(float(row["distance_m"]))
            data[sc][1].append(float(val))

fig, ax = plt.subplots(figsize=(9.2, 7.0), dpi=200)
for key, label, color, marker in scenarios:
    x, y = data[key]
    x = np.array(x); y = np.array(y)
    ax.plot(x, y, color=color, marker=marker, markersize=7,
            markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.3,
            linewidth=1.3, label=label)

ax.set_xlim(0, 500)
ax.set_ylim(-130, -20)
ax.xaxis.set_major_locator(MultipleLocator(100))
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.tick_params(axis="both", labelsize=22, direction="in", length=6, width=1.1,
               top=False, right=False)
ax.set_xlabel("\u901a\u4fe1\u8ddd\u79bb Distance /m", fontsize=25)
ax.set_ylabel("RSSI /dBm", fontsize=25)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_linewidth(1.2)
ax.legend(fontsize=19, loc="upper right", frameon=True, framealpha=0.95,
          handletextpad=0.5, borderpad=0.5)
fig.tight_layout()
fig.savefig("C:/Users/Administrator/rssi_fig3.png", bbox_inches="tight")
plt.close(fig)
print("saved rssi_fig3.png")
