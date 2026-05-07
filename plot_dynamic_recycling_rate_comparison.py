import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
SUMMARY_FILE = ROOT / "Scenario result" / "recycling_rate" / "lithium_recycling_rate_global_summary.csv"
FIGURE_DIR = ROOT / "Figure_data"

COLLECTION_ORDER = ["low_collection", "baseline", "high_collection"]
RECOVERY_ORDER = ["low", "baseline", "high"]
TECH_ORDER = ["Pyro", "Hydro", "Direct"]

LABELS = {
    "low_collection": "Low collection",
    "baseline": "Baseline",
    "high_collection": "High collection",
    "low": "Low Li recovery path",
    "high": "High Li recovery path",
    "Pyro": "Pyro",
    "Hydro": "Hydro",
    "Direct": "Direct",
}


def parse_years(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def ordered_values(values, preferred):
    values = list(values)
    ordered = [item for item in preferred if item in values]
    ordered.extend(sorted(item for item in values if item not in ordered))
    return ordered


def build_plot_data(
    summary,
    years,
    row_dimension,
    fixed_collection_scenario,
    fixed_recovery_scenario,
    primary_gap_basis,
):
    data = summary[summary["Year"].isin(years)].copy()
    if row_dimension == "collection":
        data = data[data["recovery_efficiency_scenario"] == fixed_recovery_scenario]
        row_col = "scenario"
        row_values = ordered_values(data[row_col].unique(), COLLECTION_ORDER)
    elif row_dimension == "recovery":
        data = data[data["scenario"] == fixed_collection_scenario]
        row_col = "recovery_efficiency_scenario"
        row_values = ordered_values(data[row_col].unique(), RECOVERY_ORDER)
    else:
        raise ValueError("row_dimension must be 'collection' or 'recovery'")

    data["primary_lithium_gap"] = data["collected_lithium"] - data["recycled_lithium"]
    if primary_gap_basis == "embedded":
        data["primary_lithium_gap"] = data["retired_lithium"] - data["recycled_lithium"]
    data["technology_loss"] = data["collected_lithium"] - data["recycled_lithium"]
    data["collection_loss"] = data["uncollected_lithium"]
    data["recycled_share_of_collected"] = (
        data["recycled_lithium"] / data["collected_lithium"].replace(0, pd.NA)
    ).fillna(0)
    data["recycled_share_of_embedded"] = (
            data["recycled_lithium"] / data["retired_lithium"].replace(0, pd.NA)
    ).fillna(0)
    data["gap_share_of_collected"] = (
        data["primary_lithium_gap"] / data["collected_lithium"].replace(0, pd.NA)
    ).fillna(0)
    data["gap_share_of_embedded"] = (
        data["primary_lithium_gap"] / data["retired_lithium"].replace(0, pd.NA)
    ).fillna(0)

    tech_values = ordered_values(data["recycling_m"].unique(), TECH_ORDER)
    return data, row_col, row_values, tech_values


def plot_comparison(
    data,
    row_col,
    row_values,
    tech_values,
    years,
    row_dimension,
    fixed_collection_scenario,
    fixed_recovery_scenario,
    primary_gap_basis,
    output_path,
):
    fig, axes = plt.subplots(
        len(row_values),
        len(tech_values),
        figsize=(5.2 * len(tech_values), 3.4 * len(row_values)),
        sharex=True,
        sharey=True,
    )
    if len(row_values) == 1:
        axes = [axes]
    max_li = data["retired_lithium" if primary_gap_basis == "embedded" else "collected_lithium"].max()
    y_max = max_li / 1e6 * 1.08

    for r, row_value in enumerate(row_values):
        for c, tech in enumerate(tech_values):
            ax = axes[r][c] if len(row_values) > 1 else axes[c]
            panel = data[(data[row_col] == row_value) & (data["recycling_m"] == tech)].copy()
            panel = panel.sort_values("Year")
            if panel.empty:
                ax.axis("off")
                continue

            x = panel["Year"]
            recycled = panel["recycled_lithium"] / 1e6
            gap = panel["primary_lithium_gap"] / 1e6
            potential = (
                panel["retired_lithium"] if primary_gap_basis == "embedded" else panel["collected_lithium"]
            ) / 1e6

            ax.stackplot(
                x,
                recycled,
                gap,
                colors=["#2fbf8f", "#a8afb8"],
                labels=["Recycled lithium", "Primary lithium gap"],
            )
            ax.plot(x, potential, color="#111827", linestyle="--", linewidth=1.8)
            share_col = "gap_share_of_embedded" if primary_gap_basis == "embedded" else "gap_share_of_collected"
            start_share = panel[share_col].iloc[0]
            end_share = panel[share_col].iloc[-1]
            ax.text(
                0.04,
                0.86,
                f"Gap: {start_share:.0%} -> {end_share:.0%}",
                transform=ax.transAxes,
                fontsize=9,
                color="#374151",
            )
            ax.set_ylim(0, y_max)
            ax.grid(alpha=0.25)
            if r == 0:
                ax.set_title(LABELS.get(tech, tech), fontsize=13, weight="bold")
            if c == 0:
                ax.set_ylabel(f"{LABELS.get(row_value, row_value)}\nMillion tonnes Li")
            if r == len(row_values) - 1:
                ax.set_xlabel("Year")

    if row_dimension == "collection":
        title = (
            "Collection-scenario comparison: recycled lithium vs primary lithium gap\n"
            f"Fixed technology Li recovery path: {fixed_recovery_scenario}; gap basis: {primary_gap_basis}"
        )
    else:
        title = (
            "Technology-recovery-path comparison: recycled lithium vs primary lithium gap\n"
            f"Fixed collection scenario: {fixed_collection_scenario}; gap basis: {primary_gap_basis}"
        )
    fig.suptitle(title, fontsize=17, weight="bold")

    handles = [
        plt.Line2D([0], [0], color="#2fbf8f", linewidth=8),
        plt.Line2D([0], [0], color="#a8afb8", linewidth=8),
        plt.Line2D([0], [0], color="#111827", linestyle="--", linewidth=1.8),
    ]
    labels = ["Recycled lithium", "Primary lithium gap", "Lithium potential"]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Redraw dynamic recycled-vs-primary lithium comparison with configurable comparison dimensions."
    )
    parser.add_argument("--years", default="2025,2030,2035,2040,2045,2050")
    parser.add_argument(
        "--row-dimension",
        choices=["collection", "recovery"],
        default="collection",
        help="Use collection scenarios or Li-recovery paths as rows.",
    )
    parser.add_argument("--fixed-collection-scenario", default="baseline")
    parser.add_argument("--fixed-recovery-scenario", default="baseline")
    parser.add_argument(
        "--primary-gap-basis",
        choices=["collected", "embedded"],
        default="embedded",
        help="Use collected lithium potential or total embedded lithium potential as the primary gap basis.",
    )
    parser.add_argument(
        "--output",
        default=str(FIGURE_DIR / "dynamic_recycling_rate_primary_vs_recycled_by_technology.png"),
    )
    args = parser.parse_args()

    summary = pd.read_csv(SUMMARY_FILE)
    years = parse_years(args.years)
    plot_data, row_col, row_values, tech_values = build_plot_data(
        summary,
        years,
        args.row_dimension,
        args.fixed_collection_scenario,
        args.fixed_recovery_scenario,
        args.primary_gap_basis,
    )
    output_path = Path(args.output)
    csv_path = output_path.with_suffix(".csv")
    plot_data.to_csv(csv_path, index=False)
    plot_comparison(
        plot_data,
        row_col,
        row_values,
        tech_values,
        years,
        args.row_dimension,
        args.fixed_collection_scenario,
        args.fixed_recovery_scenario,
        args.primary_gap_basis,
        output_path,
    )
    print(f"Wrote {output_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
