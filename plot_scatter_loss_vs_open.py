"""
Scatter: lithium loss (recovered Li under policy minus recovered Li under open_policy)
- x axis: current_policy loss vs open
- y axis: each comparison policy (reference/strict/critical-route) loss vs open
- Color: technology (Direct / Hydro / PyroHydro)
- Marker: 2040 triangle, 2050 star; other years light dots; PyroHydro S1-S5 spread as light cloud
"""
from pathlib import Path

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_BASE = ROOT / "unified_policy_run" / "data" / "fig3_pyrohydro_sensitivity_unified"
OUT_DIR = ROOT / "unified_policy_run" / "figures" / "fig3_pyrohydro_robustness"

SCENARIOS = ["conservative", "s2", "medium", "s4", "s5"]
BASELINE_POLICY = "current_policy"
OPEN_POLICY = "open_policy"
COMPARISON_POLICIES = ["reference_policy", "strict_policy", "critical_route_policy"]
POLICY_LABELS = {
    "reference_policy":      "Reference",
    "strict_policy":         "Strict",
    "critical_route_policy": "Critical-route",
}
TECHNOLOGY_ORDER = ["Direct", "Hydro", "PyroHydro"]
TECHNOLOGY_COLORS = {
    "Direct":    "#0072B2",
    "Hydro":     "#009E73",
    "PyroHydro": "#CC79A7",
}


def load_all():
    dfs = []
    for s in SCENARIOS:
        f = DATA_BASE / f"pyrohydro_sensitivity_{s}_unified" / "dynamic_scale_summary.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d["pyro_scenario"] = s
        dfs.append(d)
    return pd.concat(dfs, ignore_index=True)


def loss_table(data):
    """For each (pyro_scenario, year, technology, policy), compute
       loss = recovered_lithium_t[policy] - recovered_lithium_t[open_policy]
       (negative = policy recovers less than open)."""
    keep = data[data["technology"].isin(TECHNOLOGY_ORDER)].copy()
    open_df = (
        keep[keep["policy_scenario"] == OPEN_POLICY]
        [["pyro_scenario", "year", "technology", "recovered_lithium_t"]]
        .rename(columns={"recovered_lithium_t": "open_li_t"})
    )
    other_df = keep[keep["policy_scenario"] != OPEN_POLICY].copy()
    merged = other_df.merge(open_df, on=["pyro_scenario", "year", "technology"], how="left")
    merged["loss_kt"] = (merged["recovered_lithium_t"] - merged["open_li_t"]) / 1000.0
    return merged


def main():
    plt.rcParams.update({"font.family": "Arial"})
    data = load_all()
    losses = loss_table(data)
    medium = losses[losses["pyro_scenario"] == "medium"].copy()
    others = losses[losses["pyro_scenario"] != "medium"].copy()

    key_years = [2040, 2050]
    year_markers = {2040: "^", 2050: "*"}
    year_sizes   = {2040: 90,  2050: 170}

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.4), dpi=300, sharex=True, sharey=True)
    panel_labels = ["a", "b", "c"]

    # Compute combined axis limits
    def get_xy_pairs(loss_df, comparison_policy):
        cur = loss_df[loss_df["policy_scenario"] == BASELINE_POLICY][
            ["pyro_scenario", "year", "technology", "loss_kt"]
        ].rename(columns={"loss_kt": "x_kt"})
        cmp_ = loss_df[loss_df["policy_scenario"] == comparison_policy][
            ["pyro_scenario", "year", "technology", "loss_kt"]
        ].rename(columns={"loss_kt": "y_kt"})
        return cur.merge(cmp_, on=["pyro_scenario", "year", "technology"], how="inner")

    all_x, all_y = [], []
    for policy in COMPARISON_POLICIES:
        m = get_xy_pairs(losses, policy)
        all_x.extend(m["x_kt"].tolist())
        all_y.extend(m["y_kt"].tolist())
    lim = max(abs(min(all_x + all_y + [0])), abs(max(all_x + all_y + [0]))) * 1.10
    if lim == 0:
        lim = 100

    for ax, policy, panel in zip(axes, COMPARISON_POLICIES, panel_labels):
        # Light cloud (S1-S5 across all years and techs)
        m_others = get_xy_pairs(others, policy)
        for technology in TECHNOLOGY_ORDER:
            sub = m_others[m_others["technology"] == technology]
            if not sub.empty:
                ax.scatter(
                    sub["x_kt"], sub["y_kt"],
                    color=TECHNOLOGY_COLORS[technology],
                    alpha=0.18, s=14, edgecolor="none", linewidth=0,
                )
        # Medium (dark): annual trajectory dots + key year markers
        m_med = get_xy_pairs(medium, policy).sort_values("year")
        for technology in TECHNOLOGY_ORDER:
            sub = m_med[m_med["technology"] == technology].sort_values("year")
            if sub.empty:
                continue
            non_key = sub[~sub["year"].isin(key_years)]
            if not non_key.empty:
                ax.scatter(
                    non_key["x_kt"], non_key["y_kt"],
                    color=TECHNOLOGY_COLORS[technology],
                    s=22, marker="o", alpha=0.30,
                    edgecolor="none", linewidth=0, zorder=3,
                )
            for _, row in sub[sub["year"].isin(key_years)].iterrows():
                ax.scatter(
                    row["x_kt"], row["y_kt"],
                    color=TECHNOLOGY_COLORS[technology],
                    s=year_sizes[int(row["year"])],
                    marker=year_markers[int(row["year"])],
                    edgecolor="white", linewidth=0.8, zorder=4,
                )
        # Reference lines: y=x diagonal + axes
        ax.plot([-lim, lim], [-lim, lim], color="0.55", linewidth=0.9, linestyle="--", zorder=1)
        ax.axhline(0, color="0.35", linewidth=0.7, zorder=1)
        ax.axvline(0, color="0.35", linewidth=0.7, zorder=1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.text(0.02, 0.98, panel, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")
        ax.set_title(f"{POLICY_LABELS[policy]} vs Current", fontsize=11, pad=6)
        ax.set_xlabel("Current Li loss (kt)")
        ax.grid(color="0.92", linewidth=0.5)
        ax.tick_params(axis="both", labelsize=9, direction="in")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Policy Li loss (kt)")

    tech_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=TECHNOLOGY_COLORS[t],
               markeredgecolor="white", markersize=7, label=t)
        for t in TECHNOLOGY_ORDER
    ] + [
        Line2D([0], [0], marker="^", linestyle="", color="0.30",
               markeredgecolor="white", markersize=8, label="2040"),
        Line2D([0], [0], marker="*", linestyle="", color="0.30",
               markeredgecolor="white", markersize=13, label="2050"),
    ]
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.16, top=0.88, wspace=0.16)
    fig.legend(handles=tech_handles, loc="lower center", ncol=5,
               frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.02))

    fig.suptitle(
        "Lithium loss by policy (Li loss = recovered Li − recovered Li under open policy)",
        y=0.97, fontsize=12,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "scatter_loss_vs_open_combined.png"
    pdf = OUT_DIR / "scatter_loss_vs_open_combined.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
