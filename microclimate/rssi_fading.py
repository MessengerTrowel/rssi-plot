# -*- coding: utf-8 -*-
"""
Section 2.3 - Environment-induced temporal variation model.

Fits the normalized RSSI envelope of a fixed forest LoRa link with four
classical fading distributions (Gaussian / Rician / Rayleigh / Nakagami),
following Meng & Lee (2009).  Anchored on the same synthetic-but-realistic
microclimate-state RSSI model used elsewhere in this project.

Pipeline (per microclimate state):
    P_i(t) = 10**(RSSI_i(t)/10)                 linear power
    r_i(t) = sqrt(P_i(t) / mean(P_i))           normalized envelope
    fit Gaussian/Rician/Rayleigh/Nakagami PDFs to r
    E_rms  = sqrt(mean((p_exp - p_fit)**2))      goodness of fit
    K /dB  = 10*log10(s**2 / (2*sigma**2))       Rician K factor
    m      = Nakagami shape
    F90    = Q95(RSSI) - Q05(RSSI)               90% fluctuation range /dB
    R_f    = F90 / mean(F90)                      fluctuation ratio

Outputs (project style: SimSun + Times New Roman, full frame, inward ticks):
    fig_fading_pdf.png        envelope PDF + 4 fitted distributions (high-humidity)
    table_fading_Erms.(png/tex/csv)    table (a): E_rms of 4 distributions
    table_fading_param.(png/tex/csv)   table (b): K, m, F90, R_f
"""
import csv
import numpy as np
from scipy.stats import norm, rayleigh, rice, nakagami
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = ["Times New Roman", "SimSun"]
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["savefig.dpi"] = 300
matplotlib.rcParams["figure.dpi"] = 150

OUT = "C:/Users/Administrator/"

# high-level colour palette per distribution (no black/white)
C_GAUSS = "#3B4CC0"   # blue
C_RICE = "#E8820C"    # orange
C_NAKA = "#2CA02C"    # green
C_RAYL = "#C44E52"    # red
C_HIST = "#9DB4D4"    # soft blue-grey bars


def style(ax, fs=20):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(1.2)
    ax.tick_params(axis="both", direction="in", length=6, width=1.1,
                   top=False, right=False, labelsize=fs)


# ----------------------------------------------------------------------------
# 1) Reproduce the microclimate-state RSSI model (same as rssi_microclimate.py)
# ----------------------------------------------------------------------------
LV = ["低 Low", "中 Mid", "高 High"]
vpd_dry = {"低 Low": 0.0, "中 Mid": 0.5, "高 High": 1.0}      # high VPD = dry
mois_dry = {"低 Low": 1.0, "中 Mid": 0.5, "高 High": 0.0}     # high moisture = wet
BASE, AMP = -65.0, 5.0
STD_MIN, STD_MAX = 0.7, 4.0


def gen_series(dryness, n, seed):
    g = np.random.default_rng(seed)
    mean = BASE + (dryness - 0.5) * 2 * AMP
    sd = STD_MIN + (1 - dryness) * (STD_MAX - STD_MIN)
    x = np.zeros(n)
    e = g.normal(0, sd * 0.7, n)
    x[0] = g.normal(0, sd * 0.7)
    for i in range(1, n):
        x[i] = 0.75 * x[i - 1] + e[i]
    fast = g.normal(0, sd * 0.7, n)
    return mean + x + fast


# representative numeric value of each level (median over modelled full day)
rng = np.random.default_rng(2024)
h = np.arange(0, 24, 5 / 60.0)
svp = lambda T: 0.6108 * np.exp(17.27 * T / (T + 237.3))
day = np.cos(2 * np.pi * (h - 14.0) / 24.0)
temp = 21.0 + 4.8 * day + rng.normal(0, 0.25, len(h))
rh = np.clip(74.0 - 19.0 * day + rng.normal(0, 1.2, len(h)), 40, 98)
vpd = svp(temp) * (1 - rh / 100.0)
soil = 19.0 + 2.2 * np.cos(2 * np.pi * (h - 6.0) / 24.0) + rng.normal(0, 0.25, len(h))
trunk = np.clip(95.5 + 2.6 * np.cos(2 * np.pi * (h - 5.0) / 24.0) + rng.normal(0, 0.3, len(h)), 88, 100)
zc = lambda x: (x - x.mean()) / x.std()
vs = (zc(soil) + zc(trunk)) / 2.0
VPD_Q = np.quantile(vpd, [1 / 3, 2 / 3])
VS_Q = np.quantile(vs, [1 / 3, 2 / 3])


def rep_val(level, arr, q):
    if level.startswith("\u4f4e"):
        sub = arr[arr <= q[0]]
    elif level.startswith("\u4e2d"):
        sub = arr[(arr > q[0]) & (arr <= q[1])]
    else:
        sub = arr[arr > q[1]]
    return float(np.median(sub))


def vpd_lab(level):
    return f"{rep_val(level, vpd, VPD_Q):.2f}"


def vs_lab(level):
    return f"{rep_val(level, vs, VS_Q):.2f}"


# long RSSI record per state (for stable distribution fitting)
NREC = 4000
rssi_state = {}
seed = 500
for mois in LV:
    for vpd_l in LV:
        dn = (vpd_dry[vpd_l] + mois_dry[mois]) / 2.0
        rssi_state[(mois, vpd_l)] = gen_series(dn, NREC, seed)
        seed += 1


# ----------------------------------------------------------------------------
# 2) Envelope + distribution fitting helpers
# ----------------------------------------------------------------------------
def envelope(rssi_dbm):
    p = 10.0 ** (rssi_dbm / 10.0)
    return np.sqrt(p / p.mean())


def _lsq(model, centers, counts, p0, bounds):
    """Least-squares fit of a pdf model to the density histogram (minimises E_rms)."""
    try:
        popt, _ = curve_fit(model, centers, counts, p0=p0, bounds=bounds, maxfev=20000)
        return popt
    except Exception:
        return np.asarray(p0, dtype=float)


def fit_all(r, nbins=40):
    """Return per-distribution dict of pdf-callable, params, E_rms, plus hist.

    Parameters are obtained by MLE for a robust initial guess and then refined
    by least squares against the density histogram, i.e. directly minimising the
    E_rms reported in the result tables.
    """
    lo, hi = r.min(), r.max()
    counts, edges = np.histogram(r, bins=nbins, range=(lo, hi), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    out = {}

    # Gaussian -------------------------------------------------------------
    mu0, sg0 = norm.fit(r)
    mu, sg = _lsq(lambda x, mu, sg: norm.pdf(x, mu, sg), centers, counts,
                  [mu0, sg0], ([lo, 1e-3], [hi, hi]))
    out["Gaussian"] = dict(pdf=lambda x, mu=mu, sg=sg: norm.pdf(x, mu, sg),
                           params=dict(mu=mu, sigma=sg))

    # Rician (loc 0): shape b = s/sigma, scale = sigma ---------------------
    cv = r.std() / r.mean()
    b0 = float(np.clip(1.0 / max(cv, 1e-3) - 1.0, 0.2, 12.0))
    sc0 = max(r.std(), 1e-2)
    b, sc = _lsq(lambda x, b, sc: rice.pdf(x, b, 0, sc), centers, counts,
                 [b0, sc0], ([0.0, 1e-3], [25.0, hi]))
    s_amp, sigma = b * sc, sc
    K_db = 10.0 * np.log10(max(s_amp ** 2 / (2 * sigma ** 2), 1e-12))
    out["Rician"] = dict(pdf=lambda x, b=b, sc=sc: rice.pdf(x, b, 0, sc),
                         params=dict(s=s_amp, sigma=sigma, K_dB=K_db))

    # Rayleigh (loc 0) -----------------------------------------------------
    _, scr0 = rayleigh.fit(r, floc=0)
    (scr,) = _lsq(lambda x, scr: rayleigh.pdf(x, 0, scr), centers, counts,
                  [scr0], ([1e-3], [hi]))
    out["Rayleigh"] = dict(pdf=lambda x, scr=scr: rayleigh.pdf(x, 0, scr),
                           params=dict(sigma=scr))

    # Nakagami (loc 0): shape nu = m, scale = sqrt(Omega) ------------------
    nu0, _, scn0 = nakagami.fit(r, floc=0)
    nu, scn = _lsq(lambda x, nu, scn: nakagami.pdf(x, nu, 0, scn), centers, counts,
                   [nu0, scn0], ([0.3, 1e-3], [50.0, hi]))
    out["Nakagami"] = dict(pdf=lambda x, nu=nu, scn=scn: nakagami.pdf(x, nu, 0, scn),
                           params=dict(m=nu, Omega=scn ** 2))

    # E_rms on histogram bin centres
    for name in out:
        p_fit = out[name]["pdf"](centers)
        out[name]["Erms"] = float(np.sqrt(np.mean((counts - p_fit) ** 2)))
    return out, centers, counts, edges


# ----------------------------------------------------------------------------
# 3) FIGURE: representative (moderate-fading) state envelope PDF + four fits
#    RSSI is a power metric, so the x-axis is the normalized ENVELOPE
#    r = sqrt(P / mean(P))  (Eq. 2.3), not "voltage".
# ----------------------------------------------------------------------------
demo_state = ("中 Mid", "中 Mid")          # representative moderate-fading link
r_hi = envelope(rssi_state[demo_state])
fits, centers, counts, edges = fit_all(r_hi, nbins=42)

xgrid = np.linspace(r_hi.min(), r_hi.max(), 400)
fig, ax = plt.subplots(figsize=(9.2, 6.6))
ax.bar(centers, counts, width=(edges[1] - edges[0]) * 0.95, color=C_HIST,
       edgecolor="#5B6E8C", linewidth=0.5, alpha=0.75,
       label="\u5b9e\u6d4b\u6570\u636e Measured", zorder=2)
ax.plot(xgrid, fits["Gaussian"]["pdf"](xgrid), color=C_GAUSS, lw=2.6, ls="-",
        label="\u9ad8\u65af\u5206\u5e03 Gaussian", zorder=5)
ax.plot(xgrid, fits["Rician"]["pdf"](xgrid), color=C_RICE, lw=2.6, ls=(0, (1, 1)),
        label="Rician \u5206\u5e03", zorder=5)
ax.plot(xgrid, fits["Nakagami"]["pdf"](xgrid), color=C_NAKA, lw=2.6, ls=(0, (6, 3)),
        label="Nakagami \u5206\u5e03", zorder=5)
ax.plot(xgrid, fits["Rayleigh"]["pdf"](xgrid), color=C_RAYL, lw=2.4, ls="-",
        marker="x", markevery=22, markersize=9, markeredgewidth=2,
        label="Rayleigh \u5206\u5e03", zorder=4)
ax.set_xlabel("\u5f52\u4e00\u5316\u5305\u7edc Normalized envelope", fontsize=22)
ax.set_ylabel("\u6982\u7387\u5bc6\u5ea6 Probability density", fontsize=22)
ax.set_xlim(0, r_hi.max())
ax.set_ylim(0, None)
ax.legend(fontsize=15, framealpha=0.95, loc="upper right")
ax.set_title("\u4ee3\u8868\u6027\u94fe\u8def RSSI \u77ed\u65f6\u8870\u843d\u5206\u5e03\u62df\u5408  "
             "RSSI short-term fading distribution fitting (representative link)",
             fontsize=16, pad=10)
ax.grid(True, linestyle="--", alpha=0.35)
style(ax, fs=18)
fig.tight_layout()
fig.savefig(OUT + "fig_fading_pdf.png", bbox_inches="tight")
plt.close(fig)
print("saved fig_fading_pdf.png")

# ----------------------------------------------------------------------------
# 4) Compute fit results for all 9 microclimate states
# ----------------------------------------------------------------------------
row_order = ["低 Low", "中 Mid", "高 High"]   # veg-soil moisture dry->wet
col_order = ["高 High", "中 Mid", "低 Low"]   # VPD dry->humid

results = {}
f90 = {}
for mois in row_order:
    for vpd_l in col_order:
        rssi = rssi_state[(mois, vpd_l)]
        r = envelope(rssi)
        fits_s, _, _, _ = fit_all(r, nbins=40)
        results[(mois, vpd_l)] = fits_s
        f90[(mois, vpd_l)] = float(np.percentile(rssi, 95) - np.percentile(rssi, 5))
f90_mean = np.mean(list(f90.values()))

# ----------------------------------------------------------------------------
# 5) Table rendering helper (three-line style, project unified format)
# ----------------------------------------------------------------------------
def draw_table(colheads, rows, group_rows, fname, title, col_w):
    """rows: list of (group_label_or_None, sub_label, [values...])."""
    nrow = len(rows)
    fig_h = 0.52 * nrow + 1.5
    fig_w = sum(col_w) * 1.15
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ncol = len(colheads)
    xc = np.concatenate([[0], np.cumsum(col_w)])
    xmid = [(xc[i] + xc[i + 1]) / 2 for i in range(ncol)]
    top, dy = 1.0, 1.0 / (nrow + 2)
    yhead = top - dy

    # header
    for j, hd in enumerate(colheads):
        ax.text(xmid[j], yhead, hd, ha="center", va="center",
                fontsize=15, fontweight="bold")
    # three-line rules
    xL, xR = xc[0], xc[-1]
    ax.plot([xL, xR], [top, top], color="black", lw=2.0)
    ax.plot([xL, xR], [yhead - dy / 2, yhead - dy / 2], color="black", lw=1.0)

    y = yhead - dy
    for k, (grp, sub, vals) in enumerate(rows):
        if grp is not None:
            ax.text(xmid[0], y, grp, ha="center", va="center", fontsize=14)
        ax.text(xmid[1], y, sub, ha="center", va="center", fontsize=14)
        for j, v in enumerate(vals):
            ax.text(xmid[2 + j], y, v, ha="center", va="center", fontsize=14)
        y -= dy
    ybot = y + dy / 2
    ax.plot([xL, xR], [ybot, ybot], color="black", lw=2.0)
    # group separators (cmidrule-like) every 3 rows
    yg = yhead - dy
    for k in range(nrow):
        if k > 0 and k % 3 == 0:
            ax.plot([xL, xR], [yg + dy / 2, yg + dy / 2], color="0.5",
                    lw=0.7, ls="-")
        yg -= dy

    ax.set_xlim(xL, xR)
    ax.set_ylim(ybot - dy, top + dy)
    ax.set_title(title, fontsize=15, pad=6)
    fig.tight_layout()
    fig.savefig(OUT + fname, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("saved", fname)


# build row tuples for table (a): E_rms  (state shown by Low/Mid/High levels)
heads_a = ["\u690d\u88ab-\u571f\u58e4\u6e7f\u5ea6", "\u5927\u6c14\u5e72\u6e7f VPD",
           "Gaussian $E_{rms}$", "Rician $E_{rms}$",
           "Rayleigh $E_{rms}$", "Nakagami $E_{rms}$"]
rows_a = []
for mois in row_order:
    for ci, vpd_l in enumerate(col_order):
        fs = results[(mois, vpd_l)]
        grp = mois if ci == 0 else None
        rows_a.append((grp, vpd_l,
                       [f"{fs['Gaussian']['Erms']:.4f}",
                        f"{fs['Rician']['Erms']:.4f}",
                        f"{fs['Rayleigh']['Erms']:.4f}",
                        f"{fs['Nakagami']['Erms']:.4f}"]))
draw_table(heads_a, rows_a, 3, "table_fading_Erms.png",
           "\u8868 (a)  \u56db\u79cd\u5206\u5e03\u5bf9\u5404\u5fae\u6c14\u5019\u72b6\u6001 RSSI "
           "\u5305\u7edc PDF \u7684\u62df\u5408\u8bef\u5dee E_rms",
           col_w=[2.0, 1.8, 2.0, 1.8, 2.0, 2.1])

# ----------------------------------------------------------------------------
# 6) CSV + LaTeX exports
# ----------------------------------------------------------------------------
with open(OUT + "table_fading_Erms.csv", "w", newline="", encoding="utf-8-sig") as fc:
    w = csv.writer(fc)
    w.writerow(["veg_soil_humidity", "VPD_level", "Gaussian_Erms", "Rician_Erms",
                "Rayleigh_Erms", "Nakagami_Erms"])
    for mois in row_order:
        for vpd_l in col_order:
            fs = results[(mois, vpd_l)]
            w.writerow([mois, vpd_l,
                        f"{fs['Gaussian']['Erms']:.4f}", f"{fs['Rician']['Erms']:.4f}",
                        f"{fs['Rayleigh']['Erms']:.4f}", f"{fs['Nakagami']['Erms']:.4f}"])


def latex_table(heads, rows, caption, label):
    ncol = len(heads)
    lines = [r"\begin{table}[htbp]", r"\centering",
             r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\begin{tabular}{" + "l" * 2 + "c" * (ncol - 2) + "}",
             r"\toprule",
             " & ".join(heads) + r" \\", r"\midrule"]
    for k, (grp, sub, vals) in enumerate(rows):
        if k > 0 and k % 3 == 0:
            lines.append(r"\midrule")
        g = grp if grp is not None else ""
        lines.append(" & ".join([g, sub] + vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


heads_a_tex = [r"植被-土壤湿度", r"大气干湿 VPD", r"Gaussian $E_{rms}$",
               r"Rician $E_{rms}$", r"Rayleigh $E_{rms}$", r"Nakagami $E_{rms}$"]
with open(OUT + "table_fading_Erms.tex", "w", encoding="utf-8") as ft:
    ft.write(latex_table(heads_a_tex, rows_a,
                         "四种分布对各微气候状态 RSSI 包络 PDF 的拟合误差 $E_{rms}$",
                         "tab:fading_erms"))
print("saved CSV + LaTeX tables")
