import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = (
    ROOT
    / "trans"
    / "scenario_result"
    / "high_collection"
    / "baseline"
    / "technology_choice_modes"
    / "technology_choice_mode_summary.csv"
)
RATE_DETAIL_FILE = (
    ROOT
    / "Scenario result"
    / "recycling_rate"
    / "lithium_recycling_rate_detail.csv"
)
OUTPUT_DIR = ROOT / "Figure_data" / "barrier_decomposition"


def parse_years(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_collection_rates(years, collection_scenario, recovery_scenario):
    detail = pd.read_csv(RATE_DETAIL_FILE)
    detail = detail[
        (detail["scenario"] == collection_scenario)
        & (detail["recovery_efficiency_scenario"] == recovery_scenario)
        & (detail["Year"].isin(years))
    ].copy()
    if detail.empty:
        raise ValueError("No lithium recycling-rate detail rows found.")

    # The detail table is repeated by recycling_m. Keep one country/type row so
    # collection-rate scaling is not tripled across Direct/Hydro/Pyro.
    detail = detail.drop_duplicates(["Year", "country", "type"])
    rates = (
        detail.groupby("Year", as_index=False)[["retired_lithium", "collected_lithium"]]
        .sum()
        .rename(columns={"Year": "year"})
    )
    rates["collection_rate"] = rates["collected_lithium"] / rates["retired_lithium"]
    return rates[["year", "collection_rate"]]


def load_data(strategy, choice_mode, years, collection_scenario, recovery_scenario):
    choices = pd.read_csv(INPUT)
    selected = choices[
        (choices["Strategy type"] == strategy)
        & (choices["choice_mode"] == choice_mode)
        & (choices["year"].isin(years))
    ].copy()
    if selected.empty:
        raise ValueError(f"No rows found for {strategy} / {choice_mode}.")
    selected = selected.sort_values("year")

    reference = choices[
        (choices["Strategy type"] == strategy)
        & (choices["choice_mode"] == "Optimal_lithium")
        & (choices["year"].isin(years))
    ][["year", "recycled_lithium"]].rename(
        columns={"recycled_lithium": "optimal_lithium_recycled"}
    )

    data = selected.merge(reference, on="year", how="left")
    data = data.merge(
        load_collection_rates(years, collection_scenario, recovery_scenario),
        on="year",
        how="left",
    )
    data["collection_rate"] = data["collection_rate"].fillna(1.0)

    # Keep all Figure A quantities in the same post-collection technology-choice
    # scale. Potential retired Li is inferred from the selected-mode processed
    # contained Li and the independent collection rate.
    data["embedded_li"] = data["contained_lithium"] / data["collection_rate"]
    data["li_collected"] = data["contained_lithium"]
    data["supply_chain_available_secondary_li"] = data["recycled_lithium"]
    data["collection_loss"] = data["embedded_li"] - data["li_collected"]
    data["technology_loss"] = data["contained_lithium"] - data["recycled_lithium"]
    data["economic_selection_loss"] = (
        data["optimal_lithium_recycled"] - data["recycled_lithium"]
    ).clip(lower=0)
    data["capacity_mismatch_loss"] = 0.0
    data["trade_policy_loss"] = 0.0
    return data.sort_values("year")


def add_labels(ax, bars, values, fontsize=8):
    for bar, value in zip(bars, values):
        if value <= 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:,.1f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            color="#111827",
            rotation=0,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Plot Figure A: potential retired Li, collected Li, and supply-chain available secondary Li."
    )
    parser.add_argument("--strategy", default="Strategy 3")
    parser.add_argument("--choice-mode", default="Realistic_multiobjective")
    parser.add_argument("--collection-scenario", default="high_collection")
    parser.add_argument("--recovery-scenario", default="baseline")
    parser.add_argument("--years", default="2025,2030,2035,2040,2045,2050")
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "Figure_A_lithium_flow_to_supply_chain.png"),
    )
    args = parser.parse_args()

    years = parse_years(args.years)
    data = load_data(
        args.strategy,
        args.choice_mode,
        years,
        args.collection_scenario,
        args.recovery_scenario,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_data = data[
        [
            "year",
            "embedded_li",
            "li_collected",
            "supply_chain_available_secondary_li",
            "collection_loss",
            "capacity_mismatch_loss",
            "technology_loss",
            "trade_policy_loss",
            "economic_selection_loss",
            "collection_rate",
            "contained_lithium",
            "recycled_lithium",
            "primary_lithium_gap",
        ]
    ].copy()
    kt_cols = [
        col
        for col in plot_data.columns
        if col not in ["year", "collection_rate"]
    ]
    plot_data[kt_cols] = plot_data[kt_cols] / 1000.0

    csv_output = Path(args.output).with_suffix(".csv")
    if not csv_output.is_absolute():
        csv_output = ROOT / csv_output
    plot_data.to_csv(csv_output, index=False)

    plt.rcParams["font.family"] = "Arial"
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.6),
        dpi=300,
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )

    x = np.arange(len(plot_data))
    width = 0.24
    stage_cols = [
        "embedded_li",
        "li_collected",
        "supply_chain_available_secondary_li",
    ]
    labels = [
        "Potential retired Li",
        "Collected Li",
        "Available secondary Li",
    ]
    colors = ["#9CA3AF", "#2563EB", "#16A34A"]
    offsets = [-width, 0, width]

    for col, label, color, offset in zip(stage_cols, labels, colors, offsets):
        bars = axes[0].bar(
            x + offset,
            plot_data[col],
            width=width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.4,
        )
        add_labels(axes[0], bars, plot_data[col].tolist())

    entry_rate = (
        plot_data["supply_chain_available_secondary_li"] / plot_data["embedded_li"]
    ).replace([np.inf, -np.inf], 0)
    for i, rate in enumerate(entry_rate):
        axes[0].text(
            x[i],
            plot_data.loc[plot_data.index[i], "embedded_li"] * 1.10,
            f"{rate * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#166534",
            fontweight="bold",
        )

    axes[0].set_title("A1. Lithium flow into the battery supply chain", fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(plot_data["year"].astype(str))
    axes[0].set_ylabel("Lithium (kt Li)")
    axes[0].grid(axis="y", alpha=0.22)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=1, loc="upper left")

    components = [
        "collection_loss",
        "capacity_mismatch_loss",
        "technology_loss",
        "trade_policy_loss",
        "economic_selection_loss",
        "supply_chain_available_secondary_li",
    ]
    comp_labels = [
        "Collection loss",
        "Capacity mismatch",
        "Technology loss",
        "Trade-policy loss",
        "Economic selection loss",
        "Available secondary Li",
    ]
    comp_colors = ["#D1D5DB", "#F59E0B", "#8B5CF6", "#EF4444", "#64748B", "#16A34A"]
    bottom = np.zeros(len(plot_data))
    for col, label, color in zip(components, comp_labels, comp_colors):
        values = plot_data[col].fillna(0).to_numpy()
        axes[1].bar(
            x,
            values,
            bottom=bottom,
            width=0.55,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.35,
        )
        bottom += values

    axes[1].set_title("A2. Where potential secondary Li is lost", fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(plot_data["year"].astype(str))
    axes[1].set_ylabel("Lithium (kt Li)")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")

    fig.suptitle(
        f"Figure A. Potential retired lithium to supply-chain available secondary lithium\n{args.strategy}, {args.choice_mode}",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")
    print(f"Wrote {csv_output}")


if __name__ == "__main__":
    main()
