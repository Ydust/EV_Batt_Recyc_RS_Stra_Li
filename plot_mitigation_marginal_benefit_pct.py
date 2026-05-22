from pathlib import Path

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_FILE = (
    ROOT / "unified_policy_run" / "data" / "lithium_loss_scenarios_unified"
    / "lithium_loss_scenarios_summary.csv"
)
OUT_DIR = ROOT / "unified_policy_run" / "figures" / "fig4_mitigation_marginal"

YEARS = [2030, 2040, 2050]
STRATEGY = "Strategy 3"

POLICY_ORDER = ["current_policy", "reference_policy", "strict_policy", "critical_route_policy"]
POLICY_LABELS = {
    "current_policy": "Current",
    "reference_policy": "Reference",
    "strict_policy": "Strict",
    "critical_route_policy": "Critical-route",
}
POLICY_COLORS = {
    "current_policy":        "#2C7FB8",
    "reference_policy":      "#7FB3D5",
    "strict_policy":         "#D7263D",
    "critical_route_policy": "#F4A261",
}

MITIGATION_ORDER = [
    "policy_relaxation",
    "high_recovery_efficiency",
    "high_direct_maturity",
    "capacity_expansion",
    "combined_mitigation",
    "max_lithium",
    "lithium_aware_high_price",
]
MITIGATION_LABELS = {
    "policy_relaxation":         "Policy relaxation",
    "high_recovery_efficiency":  "High recovery efficiency",
    "high_direct_maturity":      "High Direct maturity",
    "capacity_expansion":        "Capacity expansion",
    "combined_mitigation":       "Combined mitigation",
    "max_lithium":               "Max-lithium objective",
    "lithium_aware_high_price":  "Li-aware high price (×10)",
}


def load_marginal_for_year(year):
    df = pd.read_csv(DATA_FILE)
    df = df[(df["strategy"] == STRATEGY) & (df["year"] == year)].copy()
    df["available_t"] = df["recovered_lithium_t"] - df["route_access_displaced_lithium_t"]
    pivot = df.pivot_table(
        index="mitigation_scenario", columns="policy_scenario", values="available_t", aggfunc="first"
    )
    base = pivot.loc["baseline"]  # baseline available Li per policy (kt-equivalent)
    delta = pivot.subtract(base, axis=1)  # absolute t
    # Relative: % of baseline per policy
    pct = delta.divide(base.replace(0, float("nan")), axis=1) * 100.0
    return pct


def main():
    plt.rcParams.update({"font.family": "Arial"})
    deltas = {y: load_marginal_for_year(y) for y in YEARS}
    # Per-year x-axis: each year zooms to its own data range
    per_year_xmax = {
        year: max(abs(d.values.min()), abs(d.values.max())) * 1.20
        for year, d in deltas.items()
    }

    n_mit = len(MITIGATION_ORDER)
    n_pol = len(POLICY_ORDER)
    bar_h = 0.18
    y_centers = np.arange(n_mit)[::-1]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.6), dpi=300, sharey=True)
    panel_labels = ["a", "b", "c"]

    for ax_idx, (year, ax) in enumerate(zip(YEARS, axes)):
        delta = deltas[year]
        year_xmax = per_year_xmax[year]
        for i, mit in enumerate(MITIGATION_ORDER):
            y0 = y_centers[i]
            for j, pol in enumerate(POLICY_ORDER):
                value = (
                    float(delta.loc[mit, pol])
                    if (mit in delta.index and pol in delta.columns) else 0.0
                )
                y = y0 + (j - (n_pol - 1) / 2.0) * bar_h
                ax.barh(y, value, height=bar_h * 0.92,
                        color=POLICY_COLORS[pol], edgecolor="white", linewidth=0.5)
                if abs(value) >= max(0.1, year_xmax * 0.025):
                    ha = "left" if value >= 0 else "right"
                    offset = 5 if value >= 0 else -5
                    ax.annotate(f"{value:+.1f}%", xy=(value, y), xytext=(offset, 0),
                                textcoords="offset points", ha=ha, va="center",
                                fontsize=7.2, color="0.20")

        ax.axvline(0, color="0.30", linewidth=1.0)
        ax.set_xlim(-year_xmax, year_xmax)
        ax.set_yticks(y_centers)
        if ax_idx == 0:
            ax.set_yticklabels([MITIGATION_LABELS[m] for m in MITIGATION_ORDER], fontsize=10)
        ax.grid(axis="x", color="0.90", linewidth=0.6)
        ax.tick_params(axis="x", labelsize=9, direction="in")
        ax.tick_params(axis="y", labelsize=10, length=0)
        ax.set_xlabel(f"Marginal benefit (% of baseline available Li) — {year}")
        ax.text(0.015, 0.98, panel_labels[ax_idx], transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    legend_handles = [
        Patch(facecolor=POLICY_COLORS[p], edgecolor="white", label=POLICY_LABELS[p])
        for p in POLICY_ORDER
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        "Marginal benefit of mitigation on supply-chain-available Li (% of baseline, 2030 / 2040 / 2050)",
        y=0.98, fontsize=12.8, fontweight="bold",
    )
    fig.text(
        0.5, 0.055,
        "Bars show kt change vs baseline within the same policy. "
        "Available Li = recovered − route-access displaced. "
        "Negative = mitigation reduces accessible Li under that policy.",
        ha="center", fontsize=8.7, color="#374151",
    )
    fig.subplots_adjust(left=0.13, right=0.99, top=0.92, bottom=0.13, wspace=0.10)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long_rows = []
    for year, d in deltas.items():
        df = d.reset_index().melt(id_vars="mitigation_scenario",
                                   var_name="policy_scenario",
                                   value_name="marginal_available_li_kt")
        df["year"] = year
        long_rows.append(df)
    pd.concat(long_rows, ignore_index=True).to_csv(
        OUT_DIR / "mitigation_marginal_benefit_pct_2030_2040_2050.csv", index=False
    )
    png = OUT_DIR / "mitigation_marginal_benefit_pct_2030_2040_2050.png"
    pdf = OUT_DIR / "mitigation_marginal_benefit_pct_2030_2040_2050.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
