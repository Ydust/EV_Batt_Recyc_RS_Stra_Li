"""
Net cumulative Li loss by policy with S1-S5 PyroHydro sensitivity band.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_BASE = ROOT / "unified_policy_run" / "data" / "fig3_pyrohydro_sensitivity_unified"
OUT_DIR = ROOT / "unified_policy_run" / "figures" / "fig3_pyrohydro_robustness"

SCENARIOS = ["conservative", "s2", "medium", "s4", "s5"]
CENTRAL_SCENARIO = "medium"
POLICY_ORDER = ["current_policy", "reference_policy", "strict_policy", "critical_route_policy"]
POLICY_LABELS = {
    "current_policy":        "Current",
    "reference_policy":      "Reference",
    "strict_policy":         "Strict",
    "critical_route_policy": "Critical-route",
}
POLICY_COLORS = {
    "current_policy":        "#2C7FB8",
    "reference_policy":      "#7FB3D5",
    "strict_policy":         "#D7263D",
    "critical_route_policy": "#F4A261",
}
REFERENCE_KEY = "open_policy"


def cumulative_loss_per_scenario(scenario):
    f = DATA_BASE / f"pyrohydro_sensitivity_{scenario}_unified" / "dynamic_scale_summary.csv"
    df = pd.read_csv(f)
    totals_t = df.groupby(["year", "policy_scenario"])["recovered_lithium_t"].sum().unstack()
    open_total = totals_t[REFERENCE_KEY]
    annual_loss_t = totals_t.subtract(open_total, axis=0)
    return annual_loss_t.cumsum() / 1000.0  # kt


def main():
    plt.rcParams.update({"font.family": "Arial"})
    cum_by_scenario = {s: cumulative_loss_per_scenario(s) for s in SCENARIOS}
    central = cum_by_scenario[CENTRAL_SCENARIO]
    years = central.index

    fig, ax = plt.subplots(figsize=(10.0, 5.6), dpi=300)
    for policy in POLICY_ORDER:
        if policy not in central.columns:
            continue
        # Compute S1-S5 band (min/max)
        per_scenario = pd.concat(
            [cum_by_scenario[s][policy] for s in SCENARIOS], axis=1
        )
        per_scenario.columns = SCENARIOS
        band_min = per_scenario.min(axis=1)
        band_max = per_scenario.max(axis=1)
        ax.fill_between(years, band_min, band_max,
                        color=POLICY_COLORS[policy], alpha=0.18, linewidth=0)
        ax.plot(years, central[policy],
                color=POLICY_COLORS[policy], linewidth=2.4,
                marker="o", markersize=4,
                markeredgecolor="white", markeredgewidth=0.6,
                label=POLICY_LABELS[policy])
        final = central[policy].iloc[-1]
        final_lo = band_min.iloc[-1]
        final_hi = band_max.iloc[-1]
        ax.annotate(
            f"{final:+.0f} kt\n[{final_lo:.0f}, {final_hi:.0f}]",
            xy=(years[-1], final),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=8.5, color=POLICY_COLORS[policy],
            fontweight="bold", va="center",
        )

    ax.axhline(0, color="0.30", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Year")
    ax.set_ylabel("Net cumulative Li loss vs open policy (kt)")
    ax.set_title(
        "Net cumulative lithium loss by policy with S1–S5 PyroHydro sensitivity band",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.grid(color="0.92", linewidth=0.5)
    ax.set_xlim(2025, 2052)
    ax.set_xticks([2025, 2030, 2035, 2040, 2045, 2050])
    ax.tick_params(axis="both", labelsize=9, direction="in")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=10)

    fig.text(
        0.5, 0.005,
        "Solid lines: PyroHydro medium configuration; shaded bands: S1–S5 PyroHydro sensitivity range. "
        "Net loss = recovered Li − recovered Li under open policy, cumulative over years.",
        ha="center", fontsize=8.7, color="#374151",
    )
    fig.subplots_adjust(left=0.10, right=0.94, top=0.90, bottom=0.14)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long_rows = []
    for s, df in cum_by_scenario.items():
        d = df.reset_index().melt(id_vars="year", var_name="policy_scenario",
                                   value_name="net_cum_li_loss_kt")
        d["pyrohydro_scenario"] = s
        long_rows.append(d)
    pd.concat(long_rows, ignore_index=True).to_csv(
        OUT_DIR / "net_cumulative_li_loss_by_policy_with_band.csv", index=False
    )
    png = OUT_DIR / "net_cumulative_li_loss_by_policy_with_band.png"
    pdf = OUT_DIR / "net_cumulative_li_loss_by_policy_with_band.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
