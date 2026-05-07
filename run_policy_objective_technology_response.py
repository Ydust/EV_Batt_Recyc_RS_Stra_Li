from pathlib import Path

import pandas as pd

import run_lithium_loss_scenarios as scenarios
import scenario_transport_paths as transport_module
from joint_policy_transport_technology_optimization import (
    build_method_cost_components,
    build_method_costs,
)
from scenario_transport_paths import (
    DEFAULT_DELAY_COST_USD_PER_T_DAY,
    load_distance_matrix,
    load_inputs,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "Figure_data" / "joint_policy_technology" / "policy_objective_technology_response"
YEAR = 2050
POLICIES = [
    "reference_policy",
    "current_policy",
    "strict_policy",
    "critical_route_policy",
]
METHODS = ["Direct", "Hydro", "Pyro"]
STRATEGY = "Strategy 3"
WASTE_CLASS = "hazardous"
TREATMENT_TYPE = "recovery"
CAPACITY_EXPANSION_COUNTRIES = ["CHN", "KOR", "JPN", "USA", "IND"]

RECOVERY_FLOOR_BY_METHOD = {
    "Direct": 0.97,
    "Hydro": 0.95,
    "Pyro": 0.35,
}

EU_COUNTRIES = {
    "Austria",
    "Belgium",
    "Bulgaria",
    "Croatia",
    "Cyprus",
    "Czech Republic",
    "Denmark",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Ireland",
    "Italy",
    "Latvia",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Netherlands",
    "Poland",
    "Portugal",
    "Romania",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Sweden",
}

TARGET_REGIONS = {
    "Global": set(),
    "China": {"CHN"},
    "United States": {"USA"},
    "European Union": None,
}

OBJECTIVE_RUNS = [
    ("economic_choice_baseline", "Global", "cost_min"),
    ("global_li_max_benchmark", "Global", "max_lithium"),
    ("domestic_li_allocation_objective", "China", "max_target_lithium"),
    ("domestic_li_allocation_objective", "United States", "max_target_lithium"),
    ("domestic_li_allocation_objective", "European Union", "max_target_lithium"),
]


def route_modeled_cost(routes):
    real = routes[~routes["is_unprocessed"].astype(bool)].copy()
    return float(
        real[
            ["transport_cost", "recycling_cost", "carbon_cost", "policy_cost"]
        ].sum(axis=1).sum()
    )


def load_target_regions():
    countries = pd.read_csv(ROOT / "all_countries.csv").dropna(subset=["country", "iso3"])
    targets = dict(TARGET_REGIONS)
    targets["European Union"] = set(countries.loc[countries["country"].isin(EU_COUNTRIES), "iso3"])
    return targets


def summarize_routes(routes, policy, objective, target_region, target_iso):
    real = routes[~routes["is_unprocessed"].astype(bool)].copy()
    target = real[real["destination_iso3"].isin(target_iso)].copy()
    global_li = float(real["recovered_lithium_t"].sum())
    target_li = float(target["recovered_lithium_t"].sum())
    return {
        "year": YEAR,
        "policy_scenario": policy,
        "objective": objective,
        "target_region": target_region,
        "global_recovered_lithium_t": global_li,
        "target_recovered_lithium_t": target_li,
        "target_share_pct": target_li / global_li * 100.0 if global_li > 0 else 0.0,
        "processed_scrap_t": float(real["scrap_t"].sum()),
        "route_modeled_cost": route_modeled_cost(routes),
        "transport_cost": float(real["transport_cost"].sum()),
        "recycling_cost": float(real["recycling_cost"].sum()),
        "carbon_cost": float(real["carbon_cost"].sum()),
        "policy_cost": float(real["policy_cost"].sum()),
    }


def summarize_technology_mix(routes, policy, objective, target_region, target_iso):
    real = routes[~routes["is_unprocessed"].astype(bool)].copy()
    rows = []
    global_total = float(real["recovered_lithium_t"].sum())
    target = real[real["destination_iso3"].isin(target_iso)].copy()
    target_total = float(target["recovered_lithium_t"].sum())

    for scope, data, total in [
        ("global", real, global_total),
        ("target_region", target, target_total),
    ]:
        by_tech = data.groupby("technology")["recovered_lithium_t"].sum()
        for technology in METHODS:
            value = float(by_tech.get(technology, 0.0))
            rows.append(
                {
                    "year": YEAR,
                    "policy_scenario": policy,
                    "objective": objective,
                    "target_region": target_region,
                    "scope": scope,
                    "technology": technology,
                    "recovered_lithium_t": value,
                    "technology_share_pct": value / total * 100.0 if total > 0 else 0.0,
                }
            )
    return rows


def run():
    transport_module.POLICY_FILE = scenarios.POLICY_FILE
    _, capacity, producer_iso, country_meta = load_inputs("high_collection")
    scrap_by_type = scenarios.load_scrap_by_type("high_collection")
    countries = pd.read_csv(ROOT / "all_countries.csv")
    distance = load_distance_matrix()
    li_content = scenarios.load_li_content()
    base_recovery = scenarios.load_recovery_efficiency("baseline", [YEAR], METHODS)
    recovery = scenarios.apply_recovery_floor_by_method(
        base_recovery, [YEAR], RECOVERY_FLOOR_BY_METHOD
    )
    emission = scenarios.load_emission_factor()
    lithium_price = scenarios.load_lithium_price("baseline")
    target_regions = load_target_regions()

    supply = scenarios.make_supply_by_type(scrap_by_type, countries, YEAR, STRATEGY)
    destination_capacity = scenarios.make_capacity(capacity, producer_iso, YEAR)
    destination_capacity = scenarios.expand_key_capacity(
        destination_capacity, CAPACITY_EXPANSION_COUNTRIES, 1.25
    )

    route_frames = []
    summary_rows = []
    mix_rows = []

    for policy in POLICIES:
        method_costs = build_method_costs(
            distance,
            destination_capacity,
            country_meta,
            METHODS,
            policy,
            YEAR,
            WASTE_CLASS,
            TREATMENT_TYPE,
            DEFAULT_DELAY_COST_USD_PER_T_DAY,
        )
        method_components = build_method_cost_components(
            distance,
            destination_capacity,
            country_meta,
            METHODS,
            policy,
            YEAR,
            WASTE_CLASS,
            TREATMENT_TYPE,
            DEFAULT_DELAY_COST_USD_PER_T_DAY,
        )
        for objective, target_region, solver_objective in OBJECTIVE_RUNS:
            target_iso = target_regions[target_region]
            routes = scenarios.solve_joint_scenario(
                supply,
                destination_capacity,
                method_costs,
                METHODS,
                method_components,
                YEAR,
                li_content,
                recovery,
                lithium_price,
                objective=solver_objective,
                target_destinations=target_iso,
            )
            routes = scenarios.add_lithium_outputs(routes, YEAR, recovery, li_content, emission)
            routes["year"] = YEAR
            routes["policy_scenario"] = policy
            routes["objective"] = objective
            routes["target_region"] = target_region
            route_frames.append(routes)
            summary_rows.append(
                summarize_routes(routes, policy, objective, target_region, target_iso)
            )
            mix_rows.extend(
                summarize_technology_mix(routes, policy, objective, target_region, target_iso)
            )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    routes_output = OUTPUT_DIR / "policy_objective_technology_response_routes.csv"
    summary_output = OUTPUT_DIR / "policy_objective_technology_response_summary.csv"
    mix_output = OUTPUT_DIR / "policy_objective_technology_response_mix.csv"
    pd.concat(route_frames, ignore_index=True).to_csv(routes_output, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_output, index=False)
    pd.DataFrame(mix_rows).to_csv(mix_output, index=False)
    return routes_output, summary_output, mix_output


def main():
    routes_output, summary_output, mix_output = run()
    print(f"Wrote {routes_output}")
    print(f"Wrote {summary_output}")
    print(f"Wrote {mix_output}")


if __name__ == "__main__":
    main()
