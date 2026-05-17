#!/usr/bin/env python3
"""
Post-ICI AKI x SDoH - Publication Figures (Nature-style, JAMA Network Open)

ALL numbers read from CSVs produced by the pipeline. Nothing hardcoded.

Usage:  python 05_figures.py
Inputs: results/{ici_aki,inpc}/00_consort_numbers.csv
        results/{ici_aki,inpc}/table1_characteristics.csv
        results/ici_aki/{insurance,income,education,employment,
                         housing,housing_stability}_coefficients.csv
        results/ici_aki/race_attenuation.csv
        results/inpc/base_coefficients.csv
        results/inpc/sensitivity_summary_comparison.csv
Output: results/figures/figure{1,2,3}_*.{pdf,png}
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

# -- Nature rcParams --
for k, v in {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.size": 7,
    "axes.labelsize": 7,
    "axes.titlesize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "legend.title_fontsize": 7,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "legend.frameon": False,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "figure.constrained_layout.use": True,
}.items():
    mpl.rcParams[k] = v
mpl.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

WONG = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "purple": "#CC79A7",
}
SIG_COLOR, NS_COLOR = WONG["blue"], "#999999"
OUTDIR = "results/figures"
os.makedirs(OUTDIR, exist_ok=True)


def save_fig(fig, name):
    fig.savefig(os.path.join(OUTDIR, f"{name}.pdf"), format="pdf")
    fig.savefig(os.path.join(OUTDIR, f"{name}.png"), format="png", dpi=600)
    print(f"  Saved: {name}.pdf + .png")


def panel_label(ax, label, x=-0.08, y=1.06):
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="right",
    )


def fmt_n(n):
    try:
        return f"{int(float(n)):,}"
    except (ValueError, TypeError):
        return str(n)


def load_consort(cohort_dir):
    fp = os.path.join(cohort_dir, "00_consort_numbers.csv")
    if not os.path.exists(fp):
        print(f"  ERROR: {fp} not found")
        return None
    df = pd.read_csv(fp)
    return dict(zip(df["step"], df["n"].astype(int)))


def load_matched_n(cohort_dir):
    fp = os.path.join(cohort_dir, "table1_characteristics.csv")
    if not os.path.exists(fp):
        return None, None
    tbl = pd.read_csv(fp, dtype=str)
    n_row = tbl[tbl.iloc[:, 0].str.strip() == "N"]
    if len(n_row) == 0:
        return None, None
    r = n_row.iloc[0]
    return r.get("Cases_N"), r.get("Controls_N")


# ================================================================
# FIGURE 1: DUAL CONSORT
# ================================================================
def figure1_consort():
    print("\n--- Figure 1: CONSORT flowchart ---")
    aou_c = load_consort("results/ici_aki")
    inpc_c = load_consort("results/inpc")
    if aou_c is None or inpc_c is None:
        print("  Skipping — need both CONSORT CSVs.")
        return
    aou_mc, aou_mt = load_matched_n("results/ici_aki")
    inpc_mc, inpc_mt = load_matched_n("results/inpc")

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.205, 8.5))
    cfgs = [
        (ax_a, aou_c, "All of Us (Discovery)", aou_mc, aou_mt, True),
        (ax_b, inpc_c, "INPC (Transportability)", inpc_mc, inpc_mt, False),
    ]
    for ax, c, title, mc, mt, has_basics in cfgs:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 20)
        ax.axis("off")
        ax.set_title(title, fontsize=7, fontweight="bold", pad=4)
        cx = 5

        def box(x, y, w, h, text, col="#E8F0FE"):
            ax.add_patch(
                FancyBboxPatch(
                    (x - w / 2, y - h / 2),
                    w,
                    h,
                    boxstyle="round,pad=0.15",
                    fc=col,
                    ec="#333333",
                    lw=0.5,
                )
            )
            ax.text(x, y, text, ha="center", va="center", fontsize=5.5, linespacing=1.3)

        def arr(x1, y1, x2, y2):
            ax.annotate(
                "",
                xy=(x2, y2),
                xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle="->", color="#333333", lw=0.6, shrinkA=2, shrinkB=2
                ),
            )

        def ebox(x, y, w, h, text):
            box(x, y, w, h, text, col="#FFF3E0")

        tk = [k for k in c if k.startswith("total_")]
        tn = fmt_n(c[tk[0]]) if tk else "?"

        y = 19.0
        box(cx, y, 5.5, 1.0, f"Total participants\nn = {tn}")
        arr(cx, y - 0.5, cx, y - 1.3)
        y = 17.0
        box(cx, y, 5.5, 1.0, f"ICI-treated\nn = {fmt_n(c.get('ici_treated','?'))}")
        arr(cx, y - 0.5, cx, y - 1.3)
        y = 15.0
        box(cx, y, 5.5, 1.0, f"ICI + cancer\nn = {fmt_n(c.get('ici_cancer','?'))}")
        arr(cx, y - 0.5, cx, y - 1.3)

        if has_basics and "ici_cancer_basics" in c:
            y = 13.2
            box(cx, y, 5.5, 1.0, f"Basics Survey\nn = {fmt_n(c['ici_cancer_basics'])}")
            en = c.get("excluded_no_basics", 0)
            if en > 0:
                ebox(8.5, 14.1, 2.8, 0.7, f"No survey\nn = {fmt_n(en)}")
                arr(cx + 2.75, 14.5, 8.5, 14.45)
            arr(cx, y - 0.5, cx, y - 1.3)
            y_next = 11.2
        else:
            y_next = 13.2

        y = y_next
        box(cx, y, 5.5, 1.0, f"Baseline Cr\nn = {fmt_n(c.get('has_baseline_cr','?'))}")
        eb = c.get("excluded_no_baseline", 0)
        if eb > 0:
            ebox(8.5, y + 0.9, 2.8, 0.7, f"No baseline\nn = {fmt_n(eb)}")
            arr(cx + 2.75, y + 0.5, 8.5, y + 0.55)
        arr(cx, y - 0.5, cx, y - 1.3)

        y -= 2.0
        box(cx, y, 5.5, 1.0, f"Eligible\nn = {fmt_n(c.get('eligible','?'))}")
        ee = c.get("excluded_eskd", 0)
        if ee > 0:
            ebox(8.5, y + 0.9, 2.8, 0.7, f"ESKD excl.\nn = {fmt_n(ee)}")
            arr(cx + 2.75, y + 0.5, 8.5, y + 0.55)
        arr(cx, y - 0.5, cx, y - 1.3)

        y -= 2.0
        box(
            cx - 1.8,
            y,
            2.5,
            1.0,
            f"Cases\nn = {fmt_n(c.get('cases','?'))}",
            col="#FFCDD2",
        )
        box(
            cx + 1.8,
            y,
            2.5,
            1.0,
            f"Controls\nn = {fmt_n(c.get('controls','?'))}",
            col="#C8E6C9",
        )
        arr(cx - 0.5, y + 1.5, cx - 1.8, y + 0.5)
        arr(cx + 0.5, y + 1.5, cx + 1.8, y + 0.5)
        arr(cx, y - 0.5, cx, y - 1.0)

        y -= 1.8
        box(
            cx, y, 5.5, 1.0, "1:4 PSM\n(NN, replacement, 0.2 SD caliper)", col="#F3E5F5"
        )
        arr(cx, y - 0.5, cx, y - 1.3)

        y -= 2.0
        box(
            cx - 1.8,
            y,
            2.5,
            1.0,
            f"Matched cases\nn = {fmt_n(mc) if mc else '?'}",
            col="#FFCDD2",
        )
        box(
            cx + 1.8,
            y,
            2.5,
            1.0,
            f"Matched controls\nn = {fmt_n(mt) if mt else '?'}",
            col="#C8E6C9",
        )
        arr(cx - 0.5, y + 1.5, cx - 1.8, y + 0.5)
        arr(cx + 0.5, y + 1.5, cx + 1.8, y + 0.5)

    panel_label(ax_a, "a", x=-0.02, y=1.02)
    panel_label(ax_b, "b", x=-0.02, y=1.02)
    save_fig(fig, "figure1_consort")
    plt.close(fig)


# ================================================================
# FIGURE 2: SDoH FOREST
# ================================================================
def figure2_sdoh_forest():
    print("\n--- Figure 2: SDoH forest plot ---")
    RES = "results/ici_aki"
    domains = [
        (
            "Insurance",
            "insurance_coefficients.csv",
            [
                ("f.insuranceMedicare", "Medicare"),
                ("f.insuranceMedicaid", "Medicaid"),
                ("f.insuranceVA_Military", "VA / Military"),
                ("f.insuranceUninsured", "Uninsured"),
                ("f.insuranceOther", "Other"),
                ("f.insuranceUnknown", "Unknown"),
            ],
        ),
        (
            "Income",
            "income_coefficients.csv",
            [
                ("f.income75k_100k", "$75K\u2013100K"),
                ("f.income50k_75k", "$50K\u201375K"),
                ("f.income25k_50k", "$25K\u201350K"),
                ("f.income10k_25k", "$10K\u201325K"),
                ("f.incomelt10k", "<$10K"),
                ("f.incomeUnknown", "Unknown"),
            ],
        ),
        (
            "Education",
            "education_coefficients.csv",
            [
                ("f.educationCollege", "College"),
                ("f.educationSome_College", "Some college"),
                ("f.educationHS_GED", "HS / GED"),
                ("f.educationlt_HS", "<High school"),
                ("f.educationUnknown", "Unknown"),
            ],
        ),
        (
            "Employment",
            "employment_coefficients.csv",
            [
                ("f.employmentSelf_Employed", "Self-employed"),
                ("f.employmentRetired", "Retired"),
                ("f.employmentUnable_to_Work", "Unable to work"),
                ("f.employmentUnemployed", "Unemployed"),
                ("f.employmentStudent", "Student"),
                ("f.employmentHomemaker", "Homemaker"),
                ("f.employmentUnknown", "Unknown"),
            ],
        ),
        (
            "Housing",
            "housing_coefficients.csv",
            [
                ("f.housingRent", "Rent"),
                ("f.housingOther_Arrangement", "Other arrangement"),
                ("f.housingUnknown", "Unknown"),
            ],
        ),
        (
            "Housing\nstability",
            "housing_stability_coefficients.csv",
            [("f.stabilityUnstable", "Unstable"), ("f.stabilityUnknown", "Unknown")],
        ),
    ]
    all_rows = []
    for dlabel, csvname, terms in domains:
        fp = os.path.join(RES, csvname)
        if not os.path.exists(fp):
            print(f"  WARNING: {fp} not found")
            continue
        df = pd.read_csv(fp)
        for vn, dl in terms:
            r = df[df["variable"] == vn]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            aor, lo, hi, p = r["exp_coef"], r["lower95"], r["upper95"], r["p"]
            if pd.isna(aor) or aor < 1e-5 or aor > 1e5:
                continue
            all_rows.append(
                dict(domain=dlabel, label=dl, aor=aor, lo=lo, hi=hi, p=p, sig=p < 0.05)
            )
    if not all_rows:
        print("  No data — skipping.")
        return
    pdf = pd.DataFrame(all_rows)
    fig_h = max(4.5, 0.28 * len(pdf) + 1.5)
    fig, ax = plt.subplots(figsize=(7.205, fig_h))
    ypos, dspans, y, prev = [], {}, 0, None
    for _, row in pdf.iterrows():
        if row["domain"] != prev:
            if prev is not None:
                y += 0.6
            dspans[row["domain"]] = [y, y]
            prev = row["domain"]
        ypos.append(y)
        dspans[row["domain"]][1] = y
        y += 1
    ax.axvline(x=1.0, color="#666666", ls="--", lw=0.5, zorder=0)
    for i, (_, row) in enumerate(pdf.iterrows()):
        c = SIG_COLOR if row["sig"] else NS_COLOR
        m = "s" if row["sig"] else "o"
        ms = 5 if row["sig"] else 3.5
        lo_d, hi_d = max(row["lo"], 0.05), min(row["hi"], 20.0)
        ax.errorbar(
            row["aor"],
            ypos[i],
            xerr=[[row["aor"] - lo_d], [hi_d - row["aor"]]],
            fmt=m,
            color=c,
            ecolor=c,
            elinewidth=0.8,
            capsize=2,
            capthick=0.5,
            markersize=ms,
            zorder=3,
        )
        ps = f"P = {row['p']:.3f}" if row["p"] >= 0.001 else "P < .001"
        ax.text(
            hi_d + 0.3,
            ypos[i],
            f"{row['aor']:.2f} ({lo_d:.2f}\u2013{hi_d:.2f})  {ps}",
            va="center",
            fontsize=5.5,
            color=c,
        )
    ax.set_yticks(ypos)
    ax.set_yticklabels(pdf["label"].tolist(), fontsize=6)
    ax.set_xlabel("Adjusted odds ratio (95% CI)")
    ax.set_xscale("log")
    ax.set_xlim(0.1, 25)
    ax.set_xticks([0.1, 0.25, 0.5, 1, 2, 4, 8])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    for d, (ymn, ymx) in dspans.items():
        ax.text(
            -0.02,
            (ymn + ymx) / 2,
            d,
            transform=mpl.transforms.blended_transform_factory(
                ax.transAxes, ax.transData
            ),
            fontsize=6.5,
            fontweight="bold",
            va="center",
            ha="right",
        )
    ax.set_title(
        "SDoH domain-specific models: AoU cohort (reference categories omitted)",
        fontsize=7,
        fontweight="bold",
        loc="left",
    )
    sh = mpl.lines.Line2D(
        [], [], color=SIG_COLOR, marker="s", ls="None", ms=5, label="P < .05"
    )
    nh = mpl.lines.Line2D(
        [], [], color=NS_COLOR, marker="o", ls="None", ms=3.5, label="P \u2265 .05"
    )
    ax.legend(handles=[sh, nh], loc="lower right", fontsize=6)
    save_fig(fig, "figure2_sdoh_forest")
    plt.close(fig)


# ================================================================
# FIGURE 3: RACE ATTENUATION + INPC SENSITIVITY
# ================================================================
def figure3_race_sensitivity():
    print("\n--- Figure 3: Race attenuation + sensitivity ---")
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(7.205, 3.2), gridspec_kw={"width_ratios": [1, 1.3]}
    )
    # Panel A
    af = "results/ici_aki/race_attenuation.csv"
    if os.path.exists(af):
        at = pd.read_csv(af)
        br = at[at["model"] == "base"].iloc[0]
        jr = at[at["model"] == "joint_sdoh"].iloc[0]
        aors = [br["black_aor"], jr["black_aor"]]
        los = [br["black_lower"], jr["black_lower"]]
        his = [br["black_upper"], jr["black_upper"]]
        apct = jr["attenuation_pct"]
        cols = [WONG["vermillion"], WONG["blue"]]
        ax_a.axhline(y=1.0, color="#666666", ls="--", lw=0.5, zorder=0)
        for i, (x, aor, lo, hi, c) in enumerate(zip([0, 1], aors, los, his, cols)):
            ax_a.errorbar(
                x,
                aor,
                yerr=[[aor - lo], [hi - aor]],
                fmt="s",
                color=c,
                ecolor=c,
                elinewidth=1.0,
                capsize=4,
                capthick=0.6,
                markersize=7,
                zorder=3,
            )
            ax_a.text(
                x,
                hi + 0.15,
                f"{aor:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                color=c,
            )
        ax_a.annotate(
            "",
            xy=(0.85, aors[1]),
            xytext=(0.15, aors[0]),
            arrowprops=dict(
                arrowstyle="->",
                color=WONG["green"],
                lw=1.5,
                connectionstyle="arc3,rad=-0.2",
            ),
        )
        ax_a.text(
            0.5,
            (aors[0] + aors[1]) / 2 + 0.25,
            f"{apct:+.0f}%\nattenuation",
            ha="center",
            va="bottom",
            fontsize=6,
            color=WONG["green"],
            fontweight="bold",
        )
        ax_a.set_xticks([0, 1])
        ax_a.set_xticklabels(["Base\nmodel", "Joint SDoH\nmodel"], fontsize=6)
        ax_a.set_ylabel("Black race AOR (95% CI)")
        ax_a.set_ylim(-0.2, 2.8)
        ax_a.set_xlim(-0.5, 1.5)
        ax_a.set_title(
            "AoU: Race attenuation", fontsize=7, fontweight="bold", loc="left"
        )
    else:
        ax_a.text(
            0.5,
            0.5,
            "race_attenuation.csv\nnot found",
            transform=ax_a.transAxes,
            ha="center",
            va="center",
            fontsize=7,
            color="red",
        )
    panel_label(ax_a, "a", x=-0.12, y=1.06)

    # Panel B
    rows_b = []
    bf = "results/inpc/base_coefficients.csv"
    if os.path.exists(bf):
        bdf = pd.read_csv(bf)
        br2 = bdf[bdf["variable"] == "f.raceBlack"]
        if len(br2) > 0:
            r = br2.iloc[0]
            rows_b.append(
                dict(
                    label="Primary (Cr \u22651.5\u00d7)",
                    aor=r["exp_coef"],
                    lo=r["lower95"],
                    hi=r["upper95"],
                    p=r["p"],
                )
            )
    sf = "results/inpc/sensitivity_summary_comparison.csv"
    if os.path.exists(sf):
        sdf = pd.read_csv(sf)
        lm = {
            "S1_delta_0.3": "S1: \u0394 \u22650.3 mg/dL",
            "S2_KDIGO2": "S2: KDIGO Stage 2",
            "S3_KDIGO3": "S3: KDIGO Stage 3",
            "S4_180day": "S4: 180-day window",
            "S5_mono_ICI": "S5: Mono-ICI only",
        }
        for _, r in sdf.iterrows():
            rows_b.append(
                dict(
                    label=lm.get(r["model"], r["model"]),
                    aor=r["exp_coef"],
                    lo=r["lower95"],
                    hi=r["upper95"],
                    p=r["p"],
                )
            )
    if rows_b:
        sd2 = pd.DataFrame(rows_b)
        yp = list(range(len(sd2)))[::-1]
        ax_b.axvline(x=1.0, color="#666666", ls="--", lw=0.5, zorder=0)
        for i, (_, row) in enumerate(sd2.iterrows()):
            sig = row["p"] < 0.05
            c = WONG["vermillion"] if sig else NS_COLOR
            m = "s" if sig else "o"
            ms = 5 if sig else 3.5
            hc = min(row["hi"], 10)
            ax_b.errorbar(
                row["aor"],
                yp[i],
                xerr=[[row["aor"] - row["lo"]], [hc - row["aor"]]],
                fmt=m,
                color=c,
                ecolor=c,
                elinewidth=0.8,
                capsize=2,
                capthick=0.5,
                markersize=ms,
                zorder=3,
            )
            ps = f"P = {row['p']:.3f}" if row["p"] >= 0.001 else "P < .001"
            ax_b.text(
                hc + 0.15,
                yp[i],
                f"{row['aor']:.2f} ({row['lo']:.2f}\u2013{hc:.2f})  {ps}",
                va="center",
                fontsize=5.5,
                color=c,
            )
        ax_b.set_yticks(yp)
        ax_b.set_yticklabels(sd2["label"].tolist(), fontsize=6)
        ax_b.set_xlabel("Black race AOR (95% CI)")
        ax_b.set_xscale("log")
        ax_b.set_xlim(0.5, 9)
        ax_b.set_xticks([0.5, 1, 2, 4, 8])
        ax_b.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax_b.set_title(
            "INPC: Black race AOR across thresholds",
            fontsize=7,
            fontweight="bold",
            loc="left",
        )
    else:
        ax_b.text(
            0.5,
            0.5,
            "Sensitivity data\nnot found",
            transform=ax_b.transAxes,
            ha="center",
            va="center",
            fontsize=7,
            color="red",
        )
    panel_label(ax_b, "b", x=-0.10, y=1.06)
    save_fig(fig, "figure3_race_sensitivity")
    plt.close(fig)


# ================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("POST-ICI AKI x SDoH - FIGURE GENERATION")
    print(f"  Output: {OUTDIR}/")
    print("=" * 70)
    figure1_consort()
    figure2_sdoh_forest()
    figure3_race_sensitivity()
    print("\n" + "=" * 70)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 70)
