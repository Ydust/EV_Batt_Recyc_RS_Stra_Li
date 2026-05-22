"""
Net cumulative Li loss by policy:
  net_loss = recovered_li_total[policy] - recovered_li_total[open]
  cumulative over years 2025-2050
PyroHydro medium configuration; 4 policies vs open.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_FILE = (
    ROOT / "unified_policy_run" / "data" / "fig3_pyrohydro_sensitivity_unified"
    / "pyrohydro_sensitivity_medium_unified" / "dynamic_scale_summary.csv"
)
OUT_DIR = ROOT / "unified_policy_run" / "figures" / "fig3_pyrohydro_robustness"

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


def main():
    plt.rcParams.update({"font.family": "Arial"})
    df = pd.read_csv(DATA_FILE)

    # Step 1+2: year total recovered Li per policy (sum across technologies)
    totals_t = df.groupby(["year", "policy_scenario"])["recovered_lithium_t"].sum().unstack()
    # Step 3: annual loss vs open
    open_total = totals_t[REFERENCE_KEY]
    annual_loss_t = totals_t.subtract(open_total, axis=0)
    # Step 4: cumulative + kt
    cum_kt = annual_loss_t.cumsum() / 1000.0

    fig, ax = plt.subplots(figsize=(10.0, 5.4), dpi=300)
    for policy in POLICY_ORDER:
        if policy not in cum_kt.columns:
            continue
        ax.plot(
            cum_kt.index, cum_kt[policy],
            color=POLICY_COLORS[policy], linewidth=2.4,
            marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.6,
            label=POLICY_LABELS[policy],
        )
        # Annotate 2050 final value
        final_val = cum_kt[policy].iloc[-1]
        ax.annotate(
            f"{final_val:+.0f} kt",
            xy=(cum_kt.index[-1], final_val),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9, color=POLICY_COLORS[policy],
            fontweight="bold", va="center",
        )

    ax.axhline(0, color="0.30", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Year")
    ax.set_ylabel("Net cumulative Li loss vs open policy (kt)")
    ax.set_title(
        "Net cumulative lithium loss by policy (PyroHydro medium configuration)",
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
        "Net loss = Σ recovered_li[policy] − Σ recovered_li[open_policy] (sum across techs, cumulative over years). "
        "Open policy is the unconstrained benchmark.",
        ha="center", fontsize=8.7, color="#374151",
    )
    fig.subplots_adjust(left=0.10, right=0.96, top=0.90, bottom=0.13)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cum_kt.to_csv(OUT_DIR / "net_cumulative_li_loss_by_policy.csv")
    png = OUT_DIR / "net_cumulative_li_loss_by_policy.png"
    pdf = OUT_DIR / "net_cumulative_li_loss_by_policy.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
