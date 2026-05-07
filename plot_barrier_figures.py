import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent


def parse_years(value):
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_decomposition(collection_scenario, recovery_scenario, years):
    path = (
        ROOT
        / "trans"
        / "scenario_result"
        / collection_scenario
        / recovery_scenario
        / "barrier_decomposition"
        / "lithium_barrier_decomposition.csv"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run barrier_decomposition.py first."
        )
    data = pd.read_csv(path)
    if years:
        data = data[data["year"].isin(years)].copy()
    return data


def save_bar(ax, path, title, ylabel="Lithium (t Li)"):
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(path, dpi=300)
    plt.close(ax.figure)


def plot_figures(data, output_dir, selected_mode, strategy):
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = data[
        (data["choice_mode"] == selected_mode) & (data["Strategy type"] == strategy)
    ].copy()
    if selected.empty:
        raise ValueError(f"No rows for {strategy} / {selected_mode}")

    components = [
        "collection_loss",
        "capacity_mismatch_loss",
        "technology_loss",
        "trade_policy_loss",
        "economic_selection_loss",
        "supply_chain_available_secondary_li",
    ]

    fig1 = selected[["year"] + components].set_index("year")
    fig1.to_csv(output_dir / "Figure_barrier_1_global_lithium_waterfall.csv")
    ax = fig1.plot(kind="bar", stacked=True, figsize=(10, 5))
    save_bar(ax, output_dir / "Figure_barrier_1_global_lithium_waterfall.png", "Lithium barrier decomposition")

    fig2 = selected[
        ["year", "embedded_li", "li_collected", "supply_chain_available_secondary_li"]
    ].set_index("year")
    fig2.to_csv(output_dir / "Figure_barrier_2_potential_to_available.csv")
    ax = fig2.plot(kind="line", marker="o", figsize=(9, 5))
    save_bar(ax, output_dir / "Figure_barrier_2_potential_to_available.png", "Potential, collected, and available secondary lithium")

    mode_compare = data[data["Strategy type"] == strategy].pivot_table(
        index="year",
        columns="choice_mode",
        values="supply_chain_available_secondary_li",
        aggfunc="sum",
    )
    mode_compare.to_csv(output_dir / "Figure_barrier_3_economic_selection_modes.csv")
    ax = mode_compare.plot(kind="bar", figsize=(10, 5))
    save_bar(ax, output_dir / "Figure_barrier_3_economic_selection_modes.png", "Economic selection across choice modes")

    policy = selected[
        [
            "year",
            "trade_policy_loss",
            "supply_chain_available_secondary_li",
            "total_netprofits",
            "recycling_CO2_em",
        ]
    ].set_index("year")
    policy.to_csv(output_dir / "Figure_barrier_4_trade_policy_diagnostics.csv")
    ax = policy[["trade_policy_loss", "supply_chain_available_secondary_li"]].plot(
        kind="bar", figsize=(9, 5)
    )
    save_bar(ax, output_dir / "Figure_barrier_4_trade_policy_diagnostics.png", "Trade-policy loss and available secondary lithium")

    strategy_compare = data[data["choice_mode"] == selected_mode].pivot_table(
        index="year",
        columns="Strategy type",
        values="supply_chain_available_secondary_li",
        aggfunc="sum",
    )
    strategy_compare.to_csv(output_dir / "Figure_barrier_5_strategy_entry_effect.csv")
    ax = strategy_compare.plot(kind="line", marker="o", figsize=(9, 5))
    save_bar(ax, output_dir / "Figure_barrier_5_strategy_entry_effect.png", "Supply-chain available secondary lithium by strategy")


def main():
    parser = argparse.ArgumentParser(
        description="Generate five barrier-decomposition figure datasets and PNGs."
    )
    parser.add_argument("--collection-scenario", default="high_collection")
    parser.add_argument("--recovery-scenario", default="baseline")
    parser.add_argument("--years", default="2025,2030,2035,2040,2045,2050")
    parser.add_argument("--selected-mode", default="Realistic_multiobjective")
    parser.add_argument("--strategy", default="Strategy 1")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "Figure_data" / "barrier_decomposition"),
    )
    args = parser.parse_args()

    data = load_decomposition(
        args.collection_scenario, args.recovery_scenario, parse_years(args.years)
    )
    plot_figures(data, Path(args.output_dir), args.selected_mode, args.strategy)
    print(f"Wrote barrier figures to {args.output_dir}")


if __name__ == "__main__":
    main()
