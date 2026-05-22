from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "Figure_data" / "joint_policy_technology"
OUT_DIR = BASE / "dynamic_scale_robustness"
VARIANTS = {
    "Fixed median / Dynamic scale": BASE / "dynamic_scale_cost" / "dynamic_scale_summary.csv",
    "Dynamic + capability": BASE / "dynamic_scale_cost_avail06" / "dynamic_scale_summary.csv",
    "Dynamic + capability + Direct x1.2": BASE
    / "dynamic_scale_cost_avail06_direct_x1p2"
    / "dynamic_scale_summary.csv",
}
POLICY_ORDER = ["reference_policy", "current_policy", "strict_policy", "critical_route_policy"]
POLICY_LABELS = {
    "reference_policy": "Reference",
    "current_policy": "Current",
    "strict_policy": "Strict",
    "critical_route_policy": "Critical-route",
}
TECH_COLORS = {"Direct": "#1f77b4", "Hydro": "#2ca02c", "Pyro": "#d62728"}


def load_summary():
    frames = []
    for variant, path in VARIANTS.items():
        data = pd.read_csv(path)
        data["variant"] = variant
        frames.append(data)
    data = pd.concat(frames, ignore_index=True)
    for column in ["year", "technology_share_pct", "route_modeled_cost", "recovered_lithium_t"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def write_comparison(data):
    pivot = data.pivot_table(
        index=["variant", "year", "policy_scenario", "cost_mode"],
        columns="technology",
        values="technology_share_pct",
        aggfunc="first",
        fill_value=0.0,
    ).reset_index()
    for technology in ["Direct", "Hydro", "Pyro"]:
        if technology not in pivot.columns:
            pivot[technology] = 0.0
    costs = data.groupby(
        ["variant", "year", "policy_scenario", "cost_mode"], as_index=False
    )["route_modeled_cost"].first()
    comparison = pivot.merge(costs, on=["variant", "year", "policy_scenario", "cost_mode"], how="left")
    comparison = comparison.sort_values(["variant", "year", "policy_scenario", "cost_mode"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUT_DIR / "dynamic_scale_robustness_summary.csv"
    comparison.to_csv(output, index=False)
    return comparison, output


def plot_comparison(comparison):
    plot_data = comparison[comparison["cost_mode"] != "fixed_median"].copy()
    variants = [
        "Dynamic + capability",
        "Dynamic + capability + Direct x1.2",
    ]
    years = [2030, 2040, 2050]
    fig, axes = plt.subplots(len(variants), len(years), figsize=(13, 6), sharey=True)
    for row_idx, variant in enumerate(variants):
        for col_idx, year in enumerate(years):
            ax = axes[row_idx, col_idx]
            subset = plot_data[
                (plot_data["variant"] == variant) & (plot_data["year"] == year)
            ].set_index("policy_scenario").reindex(POLICY_ORDER)
            bottom = pd.Series(0.0, index=POLICY_ORDER)
            x = range(len(POLICY_ORDER))
            for technology in ["Direct", "Hydro", "Pyro"]:
                values = subset[technology].fillna(0.0)
                ax.bar(
                    x,
                    values,
                    bottom=bottom,
                    color=TECH_COLORS[technology],
                    label=technology if row_idx == 0 and col_idx == 0 else None,
                )
                bottom = bottom + values
            ax.set_title(f"{variant}\n{year}", fontsize=9)
            ax.set_xticks(list(x))
            ax.set_xticklabels([POLICY_LABELS[p] for p in POLICY_ORDER], rotation=25, ha="right", fontsize=8)
            ax.set_ylim(0, 100)
            ax.grid(axis="y", color="0.9", linewidth=0.8)
            if col_idx == 0:
                ax.set_ylabel("Technology share (%)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Dynamic scale-cost robustness: Direct dominance depends on maturity penalty", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    png = OUT_DIR / "dynamic_scale_robustness_technology_mix.png"
    pdf = OUT_DIR / "dynamic_scale_robustness_technology_mix.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return png, pdf


def main():
    data = load_summary()
    comparison, output = write_comparison(data)
    png, pdf = plot_comparison(comparison)
    print(f"Wrote {output}")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
