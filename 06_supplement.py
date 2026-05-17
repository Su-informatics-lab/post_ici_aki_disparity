#!/usr/bin/env python3
"""
Post-ICI AKI x SDoH - Supplement (eTables 1-7, eFigures 1-2)

ALL numbers read from CSVs produced by the pipeline. Nothing hardcoded.

Usage:  python 06_supplement.py
Inputs: results/{ici_aki,inpc}/base_coefficients.csv
        results/{ici_aki,inpc}/08c_smd_balance.csv
        results/{ici_aki,inpc}/all_sensitivity_coefficients.csv
        results/{ici_aki,inpc}/sensitivity_summary_comparison.csv
        results/{ici_aki,inpc}/table1_characteristics.csv
        results/ici_aki/joint_sdoh_coefficients.csv
        results/inpc/base_coefficients.csv
Output: results/supplement/eTable{1-7}_*.csv
        results/supplement/eFigure{1,2}_*.{pdf,png}
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

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

WONG_BLUE = "#0072B2"
WONG_VERMILLION = "#D55E00"
NS_COLOR = "#999999"
OUTDIR = "results/supplement"
os.makedirs(OUTDIR, exist_ok=True)


def save_csv(df, name):
    path = os.path.join(OUTDIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"  Saved: {path} ({len(df)} rows)")


def save_fig(fig, name):
    fig.savefig(os.path.join(OUTDIR, f"{name}.pdf"), format="pdf")
    fig.savefig(os.path.join(OUTDIR, f"{name}.png"), format="png", dpi=600)
    print(f"  Saved: {name}.pdf + .png")


def fmt_coef(df):
    """Format coefficient CSV for publication."""
    out = df.copy()
    cols = {}
    cols["Variable"] = out["variable"]
    if "coef" in out.columns:
        cols["Coefficient"] = out["coef"].round(4)
    if "exp_coef" in out.columns:
        cols["AOR"] = out["exp_coef"].round(3)
    if "lower95" in out.columns and "upper95" in out.columns:
        cols["95% CI"] = out.apply(
            lambda r: (
                f"({r['lower95']:.3f}\u2013{r['upper95']:.3f})"
                if pd.notna(r["lower95"]) and pd.notna(r["upper95"])
                else "\u2014"
            ),
            axis=1,
        )
    if "se" in out.columns:
        cols["SE"] = out["se"].round(4)
    elif "se_coef" in out.columns:
        cols["SE"] = out["se_coef"].round(4)
    if "p" in out.columns:
        cols["P Value"] = out["p"].apply(
            lambda x: (
                f"{x:.4f}"
                if pd.notna(x) and x >= 0.001
                else (f"{x:.2e}" if pd.notna(x) else "\u2014")
            )
        )
    if "model" in out.columns:
        cols["Model"] = out["model"]
    return pd.DataFrame(cols)


# ================================================================
# eTABLES
# ================================================================
def etable1():
    print("\n--- eTable 1: AoU base model ---")
    fp = "results/ici_aki/base_coefficients.csv"
    if not os.path.exists(fp):
        print(f"  {fp} not found")
        return
    save_csv(fmt_coef(pd.read_csv(fp)), "eTable1_aou_base_model")


def etable2():
    print("\n--- eTable 2: INPC base model ---")
    fp = "results/inpc/base_coefficients.csv"
    if not os.path.exists(fp):
        print(f"  {fp} not found")
        return
    save_csv(fmt_coef(pd.read_csv(fp)), "eTable2_inpc_base_model")


def etable3():
    print("\n--- eTable 3: PSM balance ---")
    frames = []
    for cohort, label in [("ici_aki", "All of Us"), ("inpc", "INPC")]:
        fp = f"results/{cohort}/08c_smd_balance.csv"
        if not os.path.exists(fp):
            print(f"  {fp} not found")
            continue
        df = pd.read_csv(fp)
        df.insert(0, "Cohort", label)
        frames.append(df)
    if frames:
        save_csv(pd.concat(frames, ignore_index=True), "eTable3_psm_balance")


def etable4():
    print("\n--- eTable 4: AoU sensitivity ---")
    fp = "results/ici_aki/all_sensitivity_coefficients.csv"
    if not os.path.exists(fp):
        fp = "results/ici_aki/sensitivity_summary_comparison.csv"
    if not os.path.exists(fp):
        print("  No sensitivity data")
        return
    df = pd.read_csv(fp)
    key = [
        "f.raceBlack",
        "nci_cci_score",
        "f.cancerOther",
        "f.iciother_combo",
        "acei_arb",
        "diuretic",
        "ppi",
    ]
    if "variable" in df.columns:
        df = df[df["variable"].isin(key)].copy()
    save_csv(fmt_coef(df), "eTable4_aou_sensitivity")


def etable5():
    print("\n--- eTable 5: INPC sensitivity ---")
    fp = "results/inpc/all_sensitivity_coefficients.csv"
    if not os.path.exists(fp):
        fp = "results/inpc/sensitivity_summary_comparison.csv"
    if not os.path.exists(fp):
        print("  No sensitivity data")
        return
    df = pd.read_csv(fp)
    key = [
        "f.raceBlack",
        "nci_cci_score",
        "f.cancerOther",
        "f.iciother_combo",
        "acei_arb",
        "diuretic",
        "ppi",
        "nsaid",
    ]
    if "variable" in df.columns:
        df = df[df["variable"].isin(key)].copy()
    save_csv(fmt_coef(df), "eTable5_inpc_sensitivity")


def etable6():
    print("\n--- eTable 6: NCI-CCI prevalences ---")
    conditions = [
        "Acute MI",
        "History of MI",
        "CHF",
        "PVD",
        "CVD",
        "COPD",
        "Dementia",
        "Paralysis",
        "Diabetes (any)",
        "Diabetes (complicated)",
        "Renal disease",
        "Liver disease (mild)",
        "Liver disease (mod/severe)",
        "PUD",
        "Rheumatic disease",
        "AIDS",
    ]
    cohort_data = {}
    for cohort, label in [("ici_aki", "AoU"), ("inpc", "INPC")]:
        fp = f"results/{cohort}/table1_characteristics.csv"
        if not os.path.exists(fp):
            continue
        tbl = pd.read_csv(fp, dtype=str)
        var_col = tbl.columns[0]
        cond_map = {}
        for cond in conditions:
            match = tbl[tbl[var_col].str.strip().str.startswith(cond, na=False)]
            if len(match) > 0:
                r = match.iloc[0]
                cond_map[cond] = (r.get("All_N", "\u2014"), r.get("All_Pct", "\u2014"))
            else:
                cond_map[cond] = ("\u2014", "\u2014")
        cohort_data[label] = cond_map

    if not cohort_data:
        print("  No data")
        return
    rows = []
    for cond in conditions:
        row = {"Condition": cond}
        for label in ["AoU", "INPC"]:
            if label in cohort_data:
                n, pct = cohort_data[label].get(cond, ("\u2014", "\u2014"))
                row[f"{label} N"] = n
                row[f"{label} %"] = pct
        rows.append(row)
    save_csv(pd.DataFrame(rows), "eTable6_nci_cci_prevalences")


def etable7():
    print("\n--- eTable 7: AoU joint SDoH model ---")
    fp = "results/ici_aki/joint_sdoh_coefficients.csv"
    if not os.path.exists(fp):
        print(f"  {fp} not found")
        return
    save_csv(fmt_coef(pd.read_csv(fp)), "eTable7_aou_joint_sdoh_model")


# ================================================================
# eFIGURES
# ================================================================
def efigure1():
    print("\n--- eFigure 1: Love plot ---")
    fig, axes = plt.subplots(1, 2, figsize=(7.205, 3.0))
    for ax, cohort, title in [
        (axes[0], "ici_aki", "All of Us"),
        (axes[1], "inpc", "INPC"),
    ]:
        fp = f"results/{cohort}/08c_smd_balance.csv"
        if not os.path.exists(fp):
            ax.text(
                0.5,
                0.5,
                f"{cohort}\nnot found",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=7,
                color="red",
            )
            continue
        df = pd.read_csv(fp)
        var_col = df.columns[0]
        # Auto-detect pre/post SMD columns
        pre_col = post_col = None
        for c in df.columns:
            cl = c.lower().replace(" ", "").replace(".", "")
            if "diffun" in cl:
                pre_col = c
            if "diffadj" in cl:
                post_col = c
        if pre_col is None or post_col is None:
            # Fallback by position: Type, Diff.Un, V.Ratio.Un, Diff.Adj, V.Ratio.Adj
            if len(df.columns) >= 5:
                pre_col, post_col = df.columns[1], df.columns[3]
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Cannot parse",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="red",
                )
                continue

        variables = df[var_col].values
        pre_smd = pd.to_numeric(df[pre_col], errors="coerce").abs().values
        post_smd = pd.to_numeric(df[post_col], errors="coerce").abs().values
        yp = range(len(variables))

        ax.axvline(x=0.1, color="#999999", ls=":", lw=0.5)
        ax.axvline(x=0.0, color="#666666", ls="-", lw=0.3)
        for i in range(len(variables)):
            ax.plot(
                [pre_smd[i], post_smd[i]], [i, i], color="#CCCCCC", lw=0.4, zorder=1
            )
        ax.scatter(
            pre_smd,
            yp,
            marker="o",
            facecolors="none",
            edgecolors=WONG_VERMILLION,
            s=25,
            linewidths=0.8,
            label="Pre-matching",
            zorder=3,
        )
        ax.scatter(
            post_smd,
            yp,
            marker="s",
            color=WONG_BLUE,
            s=25,
            label="Post-matching",
            zorder=4,
        )
        ax.set_yticks(list(yp))
        ax.set_yticklabels([str(v).replace("_", " ") for v in variables], fontsize=5.5)
        ax.set_xlabel("|Standardized mean difference|")
        ax.set_title(title, fontsize=7, fontweight="bold", loc="left")
        ax.legend(loc="lower right", fontsize=5.5, markerscale=0.8)
        ax.set_xlim(-0.02, max(max(pre_smd), 0.35) + 0.05)

    axes[0].text(
        -0.12,
        1.06,
        "a",
        transform=axes[0].transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="right",
    )
    axes[1].text(
        -0.10,
        1.06,
        "b",
        transform=axes[1].transAxes,
        fontsize=8,
        fontweight="bold",
        va="top",
        ha="right",
    )
    save_fig(fig, "eFigure1_love_plot")
    plt.close(fig)


def efigure2():
    print("\n--- eFigure 2: INPC sensitivity forest ---")
    rows = []
    # Primary from base model
    bf = "results/inpc/base_coefficients.csv"
    if os.path.exists(bf):
        bdf = pd.read_csv(bf)
        br = bdf[bdf["variable"] == "f.raceBlack"]
        if len(br) > 0:
            r = br.iloc[0]
            rows.append(
                dict(
                    label="Primary (Cr \u22651.5\u00d7)",
                    aor=r["exp_coef"],
                    lo=r["lower95"],
                    hi=r["upper95"],
                    p=r["p"],
                )
            )
    # Sensitivity thresholds
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
            rows.append(
                dict(
                    label=lm.get(r["model"], r["model"]),
                    aor=r["exp_coef"],
                    lo=r["lower95"],
                    hi=r["upper95"],
                    p=r["p"],
                )
            )
    if not rows:
        print("  No data")
        return

    pdf = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(4.724, 3.0))
    yp = list(range(len(pdf)))[::-1]
    ax.axvline(x=1.0, color="#666666", ls="--", lw=0.5, zorder=0)

    for i, (_, row) in enumerate(pdf.iterrows()):
        sig = row["p"] < 0.05
        c = WONG_VERMILLION if sig else NS_COLOR
        m = "s" if sig else "o"
        ms = 6 if sig else 4
        hc = min(row["hi"], 10)
        ax.errorbar(
            row["aor"],
            yp[i],
            xerr=[[row["aor"] - row["lo"]], [hc - row["aor"]]],
            fmt=m,
            color=c,
            ecolor=c,
            elinewidth=0.8,
            capsize=2.5,
            capthick=0.5,
            markersize=ms,
            zorder=3,
        )
        ps = f"P = {row['p']:.3f}" if row["p"] >= 0.001 else "P < .001"
        ax.text(
            hc + 0.2,
            yp[i],
            f"AOR {row['aor']:.2f} ({row['lo']:.2f}\u2013{hc:.2f})  {ps}",
            va="center",
            fontsize=5,
            color=c,
        )

    ax.set_yticks(yp)
    ax.set_yticklabels(pdf["label"].tolist(), fontsize=6)
    ax.set_xlabel("Black race AOR (95% CI)")
    ax.set_xscale("log")
    ax.set_xlim(0.5, 12)
    ax.set_xticks([0.5, 1, 2, 4, 8])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_title(
        "INPC: Black race AOR across AKI severity thresholds",
        fontsize=7,
        fontweight="bold",
        loc="left",
    )

    # Dose-response annotation arrow from data
    # Find the KDIGO3 point (highest AOR) for annotation target
    kdigo3 = pdf[pdf["label"].str.contains("Stage 3")]
    if len(kdigo3) > 0:
        k3 = kdigo3.iloc[0]
        k3_idx = pdf.index.get_loc(kdigo3.index[0])
        ax.annotate(
            "Dose-response:\nAOR increases with\nAKI severity",
            xy=(k3["aor"], yp[k3_idx]),
            xytext=(6.0, yp[max(0, k3_idx - 2)]),
            fontsize=5.5,
            color=WONG_VERMILLION,
            fontstyle="italic",
            arrowprops=dict(
                arrowstyle="->",
                color=WONG_VERMILLION,
                lw=0.8,
                connectionstyle="arc3,rad=0.2",
            ),
        )

    save_fig(fig, "eFigure2_inpc_sensitivity_extended")
    plt.close(fig)


# ================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("POST-ICI AKI x SDoH - SUPPLEMENT GENERATION")
    print(f"  Output: {OUTDIR}/")
    print("=" * 70)
    etable1()
    etable2()
    etable3()
    etable4()
    etable5()
    etable6()
    etable7()
    efigure1()
    efigure2()
    print("\n" + "=" * 70)
    print("SUPPLEMENT GENERATION COMPLETE")
    print("=" * 70)
