import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RECYCLING_RATE_DIR = ROOT / "Scenario result" / "recycling_rate"
RATE_DETAIL_FILE = RECYCLING_RATE_DIR / "lithium_recycling_rate_detail.csv"
METAL_CONTENT_FILE = ROOT / "cost" / "Metal content.csv"
SCENARIO_SCRAP_FILE = (
    RECYCLING_RATE_DIR / "EV_battery_inuse_scrap_collected_by_scenario.csv"
)


def parse_years(value):
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def result_root(collection_scenario, recovery_scenario):
    return ROOT / "trans" / "scenario_result" / collection_scenario / recovery_scenario


def load_embedded_lithium(collection_scenario, recovery_scenario, years):
    if RATE_DETAIL_FILE.exists():
        detail = pd.read_csv(RATE_DETAIL_FILE)
        detail = detail[
            (detail["scenario"] == collection_scenario)
            & (detail["recovery_efficiency_scenario"] == recovery_scenario)
        ].copy()
        if years:
            detail = detail[detail["Year"].isin(years)]
        if not detail.empty:
            key_cols = ["Year", "country", "type"]
            detail = detail.drop_duplicates(key_cols)
            summary = (
                detail.groupby("Year", as_index=False)[
                    [
                        "scrap_original",
                        "scrap_collected",
                        "scrap_uncollected",
                        "retired_lithium",
                        "collected_lithium",
                        "uncollected_lithium",
                    ]
                ]
                .sum()
                .rename(
                    columns={
                        "Year": "year",
                        "retired_lithium": "embedded_li",
                        "collected_lithium": "li_collected",
                        "uncollected_lithium": "collection_loss",
                    }
                )
            )
            return summary

    if not SCENARIO_SCRAP_FILE.exists():
        raise FileNotFoundError(
            "No recycling-rate lithium detail found. Run recycling_rate_scenarios.py first."
        )
    scrap = pd.read_csv(SCENARIO_SCRAP_FILE)
    scrap = scrap[scrap["scenario"] == collection_scenario].copy()
    if years:
        scrap = scrap[scrap["Year"].isin(years)]
    metal = pd.read_csv(METAL_CONTENT_FILE).dropna(subset=["Type"])
    li_content = metal.set_index("Type")["Li"].astype(float)
    scrap["li_content"] = scrap["type"].map(li_content).fillna(0.0)
    scrap["li_embedded_original"] = scrap["scrap_original"] * scrap["li_content"]
    scrap["li_collected"] = scrap["scrap"] * scrap["li_content"]
    scrap["li_uncollected"] = scrap["scrap_uncollected"] * scrap["li_content"]
    return (
        scrap.groupby("Year", as_index=False)[
            [
                "scrap_original",
                "scrap",
                "scrap_uncollected",
                "li_embedded_original",
                "li_collected",
                "li_uncollected",
            ]
        ]
        .sum()
        .rename(
            columns={
                "Year": "year",
                "scrap": "scrap_collected",
                "li_embedded_original": "embedded_li",
                "li_uncollected": "collection_loss",
            }
        )
    )


def load_choice_mode_summary(base_root, years):
    mode_path = base_root / "technology_choice_modes" / "technology_choice_mode_summary.csv"
    global_path = base_root / "technology_choice_modes" / "technology_choice_global_summary.csv"
    if mode_path.exists():
        summary = pd.read_csv(mode_path)
    elif global_path.exists():
        summary = pd.read_csv(global_path)
        summary = (
            summary.groupby(["year", "Strategy type", "choice_mode"], as_index=False)[
                [
                    "scrap",
                    "contained_lithium",
                    "recycled_lithium",
                    "primary_lithium_gap",
                    "recycling_CO2_em",
                    "total_netprofits",
                    "total_costs",
                ]
            ]
            .sum()
            .sort_values(["year", "Strategy type", "choice_mode"])
        )
    else:
        raise FileNotFoundError(
            f"No technology-choice summary found under {base_root / 'technology_choice_modes'}"
        )
    if years:
        summary = summary[summary["year"].isin(years)].copy()
    return summary


def load_policy_reference(path, years):
    if not path:
        return pd.DataFrame()
    ref_path = Path(path)
    if ref_path.is_dir():
        ref_path = ref_path / "technology_choice_modes" / "technology_choice_mode_summary.csv"
    if not ref_path.exists():
        raise FileNotFoundError(ref_path)
    ref = pd.read_csv(ref_path)
    if years:
        ref = ref[ref["year"].isin(years)].copy()
    return ref


def build_policy_lookup(policy_reference):
    if policy_reference.empty:
        return {}
    lithium = policy_reference[
        policy_reference["choice_mode"] == "Optimal_lithium"
    ].copy()
    return {
        (int(row["year"]), row["Strategy type"]): float(row["recycled_lithium"])
        for _, row in lithium.iterrows()
    }


def decompose_barriers(
    collection_scenario,
    recovery_scenario,
    years=None,
    economic_reference_mode="Optimal_lithium",
    selected_mode="Realistic_multiobjective",
    policy_reference=None,
):
    base_root = result_root(collection_scenario, recovery_scenario)
    embedded = load_embedded_lithium(collection_scenario, recovery_scenario, years)
    choices = load_choice_mode_summary(base_root, years)
    policy_lookup = build_policy_lookup(load_policy_reference(policy_reference, years))

    rows = []
    for (year, strategy), strategy_modes in choices.groupby(["year", "Strategy type"]):
        embedded_row = embedded[embedded["year"] == year]
        if embedded_row.empty:
            continue
        embedded_row = embedded_row.iloc[0]

        reference = strategy_modes[
            strategy_modes["choice_mode"] == economic_reference_mode
        ]
        if reference.empty:
            reference = strategy_modes.sort_values("recycled_lithium", ascending=False).head(1)
        reference = reference.iloc[0]
        open_recycled = policy_lookup.get((int(year), strategy), np.nan)

        for _, selected in strategy_modes.iterrows():
            raw_reference_recycled = float(reference["recycled_lithium"])
            raw_selected_recycled = float(selected["recycled_lithium"])
            raw_reference_contained = float(reference["contained_lithium"])
            raw_selected_contained = float(selected["contained_lithium"])
            collected_li = float(embedded_row["li_collected"])
            bounded_reference_contained = min(raw_reference_contained, collected_li)
            bounded_reference_recycled = min(
                raw_reference_recycled, bounded_reference_contained
            )
            bounded_selected_recycled = min(
                raw_selected_recycled, bounded_reference_recycled
            )
            economic_loss = max(
                0.0,
                bounded_reference_recycled - bounded_selected_recycled,
            )
            trade_policy_loss = (
                max(
                    0.0,
                    min(float(open_recycled), collected_li) - bounded_reference_recycled,
                )
                if not np.isnan(open_recycled)
                else 0.0
            )
            capacity_mismatch_loss = max(
                0.0,
                collected_li - bounded_reference_contained,
            )
            technology_loss = max(
                0.0,
                bounded_reference_contained - bounded_reference_recycled,
            )
            unit_flag = (
                "raw_recycled_exceeds_collected_li"
                if raw_selected_recycled > collected_li * 1.001
                else ""
            )
            rows.append(
                {
                    "year": int(year),
                    "collection_scenario": collection_scenario,
                    "recovery_efficiency_scenario": recovery_scenario,
                    "Strategy type": strategy,
                    "choice_mode": selected["choice_mode"],
                    "economic_reference_mode": economic_reference_mode,
                    "supply_chain_boundary": (
                        "battery-supply-chain-available secondary lithium equivalent"
                    ),
                    "scrap_original": embedded_row.get("scrap_original", np.nan),
                    "scrap_collected": embedded_row.get("scrap_collected", np.nan),
                    "scrap_uncollected": embedded_row.get("scrap_uncollected", np.nan),
                    "embedded_li": float(embedded_row["embedded_li"]),
                    "li_collected": float(embedded_row["li_collected"]),
                    "collection_loss": float(embedded_row["collection_loss"]),
                    "capacity_mismatch_loss": capacity_mismatch_loss,
                    "technology_loss": technology_loss,
                    "trade_policy_loss": trade_policy_loss,
                    "economic_selection_loss": economic_loss,
                    "supply_chain_available_secondary_li": bounded_selected_recycled,
                    "contained_li_processed": min(raw_selected_contained, collected_li),
                    "raw_supply_chain_available_secondary_li": raw_selected_recycled,
                    "raw_contained_li_processed": raw_selected_contained,
                    "primary_lithium_gap": float(selected["primary_lithium_gap"]),
                    "recycling_CO2_em": float(selected.get("recycling_CO2_em", 0.0)),
                    "total_netprofits": float(selected.get("total_netprofits", 0.0)),
                    "total_costs": float(selected.get("total_costs", 0.0)),
                    "reference_recycled_li": bounded_reference_recycled,
                    "raw_reference_recycled_li": raw_reference_recycled,
                    "policy_reference_recycled_li": open_recycled,
                    "unit_consistency_flag": unit_flag,
                }
            )

    decomposition = pd.DataFrame(rows)
    output_dir = base_root / "barrier_decomposition"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "lithium_barrier_decomposition.csv"
    decomposition.to_csv(output_path, index=False)

    long_cols = [
        "collection_loss",
        "capacity_mismatch_loss",
        "technology_loss",
        "trade_policy_loss",
        "economic_selection_loss",
        "supply_chain_available_secondary_li",
    ]
    long = decomposition.melt(
        id_vars=[
            "year",
            "collection_scenario",
            "recovery_efficiency_scenario",
            "Strategy type",
            "choice_mode",
            "embedded_li",
            "li_collected",
            "supply_chain_boundary",
        ],
        value_vars=long_cols,
        var_name="barrier_component",
        value_name="lithium_t",
    )
    long.to_csv(output_dir / "lithium_barrier_decomposition_long.csv", index=False)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Decompose recycled-lithium entry barriers from collection, technology, capacity, trade policy, and economic selection."
    )
    parser.add_argument("--collection-scenario", default="high_collection")
    parser.add_argument("--recovery-scenario", default="baseline")
    parser.add_argument("--years", default="2025,2030,2035,2040,2045,2050")
    parser.add_argument("--economic-reference-mode", default="Optimal_lithium")
    parser.add_argument("--selected-mode", default="Realistic_multiobjective")
    parser.add_argument(
        "--policy-reference",
        default="",
        help="Optional open-policy technology_choice_mode_summary.csv or result root for trade-policy loss.",
    )
    args = parser.parse_args()

    output_path = decompose_barriers(
        args.collection_scenario,
        args.recovery_scenario,
        years=parse_years(args.years),
        economic_reference_mode=args.economic_reference_mode,
        selected_mode=args.selected_mode,
        policy_reference=args.policy_reference or None,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
