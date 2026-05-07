import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import scenario_transport_paths as transport_module
from scenario_transport_paths import (
    BIG_M,
    FORBIDDEN_COST,
    DEFAULT_DELAY_COST_USD_PER_T_DAY,
    apply_policy_constraints,
    build_country_groups,
    build_total_cost_matrix,
    load_distance_matrix,
    load_inputs,
    policy_route_diagnostics,
    select_policy_nodes,
    solve_transport,
    solve_transport_with_policy_fallback,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "Figure_data" / "policy_constraint_strength"
POLICY_FILE = ROOT / "waste_trade_policy_constraints.csv"


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_years(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_policy_rules(policy_scenario, year, waste_class, treatment_type):
    policy = pd.read_csv(POLICY_FILE)
    return policy[
        (policy["scenario"] == policy_scenario)
        & (policy["start_year"] <= year)
        & (policy["end_year"] >= year)
        & (policy["waste_class"].isin([waste_class, "ALL"]))
        & (policy["treatment_type"].isin([treatment_type, "ALL"]))
    ].copy()


def make_supply_demand(collection_scenario, year, strategy):
    scrap, capacity, producer_iso, country_meta = load_inputs(collection_scenario)
    year_scrap = scrap[scrap["Year"] == year].copy()
    year_capacity = capacity[capacity["Year"] == year].copy()
    demand = year_capacity[year_capacity["iso3"].isin(producer_iso)][
        ["iso3", "Mass_v"]
    ].copy()
    demand = demand.groupby("iso3", as_index=True)["Mass_v"].sum().to_frame()

    if strategy == "Strategy 2":
        supply = year_scrap[year_scrap["iso3"].isin(producer_iso)][
            ["iso3", "scrap"]
        ].copy()
    elif strategy == "Strategy 3":
        supply = year_scrap[["iso3", "scrap"]].copy()
    else:
        raise ValueError("strategy must be Strategy 2 or Strategy 3")
    supply = supply.groupby("iso3", as_index=True)["scrap"].sum().to_frame()
    return supply, demand, country_meta


def matrix_to_route_table(matrix, value_col):
    rows = matrix.stack().reset_index()
    rows.columns = ["source_iso3", "destination_iso3", value_col]
    return rows


def summarize_plan(
    plan,
    base_cost,
    policy_cost,
    policy_diag,
    supply,
    policy_scenario,
    year,
    method,
    strategy,
):
    if plan.empty:
        return {
            "year": year,
            "policy_scenario": policy_scenario,
            "method": method,
            "strategy": strategy,
            "transported_scrap_t": 0.0,
            "cross_border_scrap_t": 0.0,
            "cross_border_share": 0.0,
            "total_base_cost": 0.0,
            "total_policy_adjusted_cost": 0.0,
            "policy_cost_delta": 0.0,
            "used_forbidden_flow_t": 0.0,
            "forbidden_route_count": int((policy_diag["forbidden"] == 1).sum())
            if not policy_diag.empty
            else 0,
            "penalized_route_count": int((policy_diag["modeled_policy_cost_per_kg"] > 0).sum())
            if not policy_diag.empty
            else 0,
            "source_scrap_on_forbidden_routes_t": 0.0,
        }

    common_rows = [idx for idx in plan.index if idx in base_cost.index]
    common_cols = [col for col in plan.columns if col in base_cost.columns]
    plan = plan.loc[common_rows, common_cols].copy()
    base_cost = base_cost.loc[common_rows, common_cols].copy()
    policy_cost = policy_cost.loc[common_rows, common_cols].copy()

    total_flow = float(plan.values.sum())
    cross_border = 0.0
    used_forbidden = 0.0
    for src in plan.index:
        for dst in plan.columns:
            flow = float(plan.loc[src, dst])
            if src != dst:
                cross_border += flow
            if policy_cost.loc[src, dst] >= FORBIDDEN_COST and flow > 0:
                used_forbidden += flow

    policy_delta_matrix = (policy_cost - base_cost).clip(lower=0)
    total_base_cost = float((plan * base_cost).values.sum())
    total_policy_cost = float((plan * policy_cost.replace(FORBIDDEN_COST, np.nan)).fillna(0).values.sum())
    policy_cost_delta = float((plan * policy_delta_matrix.replace(FORBIDDEN_COST, 0)).values.sum())

    source_scrap_on_forbidden = 0.0
    if not policy_diag.empty:
        forbidden_sources = set(
            policy_diag.loc[policy_diag["forbidden"] == 1, "source_iso3"].dropna()
        )
        source_scrap_on_forbidden = float(
            supply.loc[supply.index.isin(forbidden_sources), "scrap"].sum()
        )

    return {
        "year": year,
        "policy_scenario": policy_scenario,
        "method": method,
        "strategy": strategy,
        "transported_scrap_t": total_flow,
        "cross_border_scrap_t": cross_border,
        "cross_border_share": cross_border / total_flow if total_flow else 0.0,
        "total_base_cost": total_base_cost,
        "total_policy_adjusted_cost": total_policy_cost,
        "policy_cost_delta": policy_cost_delta,
        "used_forbidden_flow_t": used_forbidden,
        "forbidden_route_count": int((policy_diag["forbidden"] == 1).sum())
        if not policy_diag.empty
        else 0,
        "penalized_route_count": int((policy_diag["modeled_policy_cost_per_kg"] > 0).sum())
        if not policy_diag.empty
        else 0,
        "source_scrap_on_forbidden_routes_t": source_scrap_on_forbidden,
    }


def analyze(
    collection_scenario,
    recovery_scenario,
    policies,
    years,
    methods,
    strategy,
    waste_class,
    treatment_type,
    delay_cost,
    policy_file=POLICY_FILE,
):
    transport_module.POLICY_FILE = Path(policy_file)
    distance = load_distance_matrix()
    rows = []
    route_rows = []
    for year in years:
        supply, demand, country_meta = make_supply_demand(collection_scenario, year, strategy)
        for method in methods:
            base_cost = build_total_cost_matrix(
                distance,
                demand,
                country_meta,
                method,
                "no_such_policy_scenario",
                year,
                waste_class,
                treatment_type,
                delay_cost,
            )
            for policy_scenario in policies:
                policy_cost = build_total_cost_matrix(
                    distance,
                    demand,
                    country_meta,
                    method,
                    policy_scenario,
                    year,
                    waste_class,
                    treatment_type,
                    delay_cost,
                )
                policy_diag = policy_route_diagnostics(
                    distance.reindex(columns=demand.index, fill_value=BIG_M),
                    country_meta,
                    policy_scenario,
                    year,
                    waste_class,
                    treatment_type,
                    delay_cost,
                )
                plan = solve_transport_with_policy_fallback(policy_cost, supply, demand)
                rows.append(
                    summarize_plan(
                        plan,
                        base_cost,
                        policy_cost,
                        policy_diag,
                        supply,
                        policy_scenario,
                        year,
                        method,
                        strategy,
                    )
                )

                route_table = plan.stack().reset_index()
                route_table.columns = ["source_iso3", "destination_iso3", "scrap_t"]
                route_table = route_table[route_table["scrap_t"] > 0].copy()
                route_table["year"] = year
                route_table["policy_scenario"] = policy_scenario
                route_table["method"] = method
                route_table["strategy"] = strategy
                route_table["cross_border"] = (
                    route_table["source_iso3"] != route_table["destination_iso3"]
                )
                route_rows.append(route_table)

    summary = pd.DataFrame(rows)
    routes = pd.concat(route_rows, ignore_index=True) if route_rows else pd.DataFrame()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "policy_constraint_strength_summary.csv"
    route_path = OUTPUT_DIR / "policy_constraint_strength_used_routes.csv"
    summary.to_csv(summary_path, index=False)
    routes.to_csv(route_path, index=False)
    return summary_path, route_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare open, baseline, and strict waste-trade policy constraints without overwriting scenario paths."
    )
    parser.add_argument("--collection-scenario", default="high_collection")
    parser.add_argument("--recovery-scenario", default="baseline")
    parser.add_argument(
        "--policies",
        default=(
            "reference_policy,current_policy,strict_policy,"
            "critical_route_policy,domestic_processing_policy"
        ),
    )
    parser.add_argument("--years", default="2025,2030,2035,2040,2045,2050")
    parser.add_argument("--methods", default="Direct,Hydro,Pyro")
    parser.add_argument("--strategy", default="Strategy 3")
    parser.add_argument("--waste-class", default="hazardous")
    parser.add_argument("--treatment-type", default="recovery")
    parser.add_argument("--delay-cost", type=float, default=DEFAULT_DELAY_COST_USD_PER_T_DAY)
    parser.add_argument("--policy-file", default=str(POLICY_FILE))
    args = parser.parse_args()

    summary_path, route_path = analyze(
        args.collection_scenario,
        args.recovery_scenario,
        parse_csv(args.policies),
        parse_years(args.years),
        parse_csv(args.methods),
        args.strategy,
        args.waste_class,
        args.treatment_type,
        args.delay_cost,
        args.policy_file,
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {route_path}")


if __name__ == "__main__":
    main()
