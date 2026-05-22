"""
Net cumulative Li loss by policy, broken down by technology.
2x2 grid of policies; each panel shows per-tech contribution + total line.
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

POLICY_ORDER = ["current_policy", "reference_policy", "critical_route_policy", "strict_policy"]
POLICY_LABELS = {
    "current_policy":        "Current",
    "reference_policy":      "Reference",
    "strict_policy":         "Strict",
    "critical_route_policy": "Critical-route",
}
TECHNOLOGY_ORDER = ["Direct", "Hydro", "Pyro", "PyroHydro"]
TECHNOLOGY_COLORS = {
    "Direct":    "#0072B2",
    "Hydro":     "#009E73",
    "Pyro":      "#8B7355",
    "PyroHydro": "#CC79A7",
}
REFERENCE_KEY = "open_policy"


def main():
    plt.rcParams.update({"font.family": "Arial"})
    df = pd.read_csv(DATA_FILE)

    # For each (year, technology, policy), compute net loss vs open
    grouped = df.groupby(["year", "technology", "policy_scenario"])["recovered_lithium_t"].sum().unstack("policy_scenario")
    open_col = grouped[REFERENCE_KEY]
    loss_t = grouped.subtract(open_col, axis=0)
    # cumulative per (technology, policy) over years
    # Reset and pivot
    loss_t = loss_t.reset_index()
    cum_records = []
    for tech in TECHNOLOGY_ORDER:
        sub = loss_t[loss_t["technology"] == tech].set_index("year").sort_index()
        for policy in POLICY_ORDER:
            if policy not in sub.columns:
                continue
            cum = sub[policy].cumsum() / 1000.0  # kt
            for year, val in cum.items():
                cum_records.append(
                    {"year": year, "technology": tech, "policy_scenario": policy, "cum_kt": val}
                )
    cum = pd.DataFrame(cum_records)

    # Also compute net total
    total_grouped = df.groupby(["year", "policy_scenario"])["recovered_lithium_t"].sum().unstack()
    total_loss = total_grouped.subtract(total_grouped[REFERENCE_KEY], axis=0)
    total_cum_kt = total_loss.cumsum() / 1000.0

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), dpi=300, sharex=True, sharey=True)
    panel_letters = [["a", "b"], ["c", "d"]]
    all_y = cum["cum_kt"].tolist() + total_cum_kt.values.flatten().tolist()
    ylim = (min(all_y) * 1.05, max(all_y + [0]) * 1.05 + 20)

    for idx, policy in enumerate(POLICY_ORDER):
        r, c = idx // 2, idx % 2
        ax = axes[r, c]
        for tech in TECHNOLOGY_ORDER:
            sub = cum[(cum["technology"] == tech) & (cum["policy_scenario"] == policy)].sort_values("year")
            if sub.empty:
                continue
            ax.plot(sub["year"], sub["cum_kt"],
                    color=TECHNOLOGY_COLORS[tech], linewidth=1.8,
                    label=tech)
        # Total line in black
        if policy in total_cum_kt.columns:
            tot = total_cum_kt[policy]
            ax.plot(tot.index, tot.values, color="black",
                    linewidth=2.4, linestyle="--", label="Net total")
            # Annotate final total
            ax.annotate(
                f"net 2050: {tot.iloc[-1]:+.0f} kt",
                xy=(tot.index[-1], tot.iloc[-1]),
                xytext=(-8, 8 if tot.iloc[-1] < 0 else -16),
                textcoords="offset points",
                fontsize=8.5, fontweight="bold", color="black",
                ha="right",
            )
        ax.axhline(0, color="0.30", linewidth=0.7, linestyle="-")
        ax.set_title(POLICY_LABELS[policy], fontsize=12, fontweight="bold", pad=6)
        ax.text(0.02, 0.97, panel_letters[r][c], transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")
        ax.set_xlim(2025, 2052)
        ax.set_xticks([2025, 2030, 2035, 2040, 2045, 2050])
        ax.set_ylim(*ylim)
        ax.grid(color="0.92", linewidth=0.4)
        ax.tick_params(axis="both", labelsize=9, direction="in")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        if r == 1:
            ax.set_xlabel("Year")
        if c == 0:
            ax.set_ylabel("Cumulative Li loss (kt)")

    # Single shared legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(
        "Net cumulative Li loss by policy, decomposed by technology (PyroHydro medium)",
        y=0.97, fontsize=13, fontweight="bold",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.91, bottom=0.10, hspace=0.20, wspace=0.10)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cum.to_csv(OUT_DIR / "net_cumulative_li_loss_by_tech.csv", index=False)
    png = OUT_DIR / "net_cumulative_li_loss_by_tech.png"
    pdf = OUT_DIR / "net_cumulative_li_loss_by_tech.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
