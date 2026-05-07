from pathlib import Path

import pandas as pd

import run_lithium_loss_scenarios as scenarios
import scenario_transport_paths as transport_module
from joint_policy_transport_technology_optimization import build_method_cost_components, build_method_costs
from scenario_transport_paths import DEFAULT_DELAY_COST_USD_PER_T_DAY, load_distance_matrix, load_inputs


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "Figure_data" / "joint_policy_technology" / "objective_benchmarks"
YEAR = 2050
POLICY = "reference_policy"
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
    "China": {"CHN"},
    "United States": {"USA"},
    "European Union": None,
}


def route_modeled_cost(routes):
    real = routes[~routes["is_unprocessed"].astype(bool)].copy()
    return float(
        real[
            ["transport_cost", "recycling_cost", "carbon_cost", "policy_cost"]
        ].sum(axis=1).sum()
    )


def load_target_regions():
    countries = pd.read_csv(ROOT / "all_countries.csv").dropna(subset=["country", "iso3"])
    eu_iso = set(countries.loc[countries["country"].isin(EU_COUNTRIES), "iso3"])
    targets = dict(TARGET_REGIONS)
    targets["European Union"] = eu_iso
    return targets


def summarize_routes(routes, objective, target_region, target_iso):
    real = routes[~routes["is_unprocessed"].astype(bool)].copy()
    target = real[real["destination_iso3"].isin(target_iso)].copy()
    return {
        "year": YEAR,
        "policy_scenario": POLICY,
        "objective": objective,
        "target_region": target_region,
        "global_recovered_lithium_t": float(real["recovered_lithium_t"].sum()),
        "target_recovered_lithium_t": float(target["recovered_lithium_t"].sum()),
        "target_share_pct": (
            float(target["recovered_lithium_t"].sum())
            / float(real["recovered_lithium_t"].sum())
            * 100.0
            if float(real["recovered_lithium_t"].sum()) > 0
            else 0.0
        ),
        "processed_scrap_t": float(real["scrap_t"].sum()),
        "route_modeled_cost": route_modeled_cost(routes),
        "transport_cost": float(real["transport_cost"].sum()),
        "recycling_cost": float(real["recycling_cost"].sum()),
        "carbon_cost": float(real["carbon_cost"].sum()),
        "policy_cost": float(real["policy_cost"].sum()),
    }


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

    supply = scenarios.make_supply_by_type(scrap_by_type, countries, YEAR, STRATEGY)
    destination_capacity = scenarios.make_capacity(capacity, producer_iso, YEAR)
    destination_capacity = scenarios.expand_key_capacity(
        destination_capacity, CAPACITY_EXPANSION_COUNTRIES, 1.25
    )
    method_costs = build_method_costs(
        distance,
        destination_capacity,
        country_meta,
        METHODS,
        POLICY,
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
        POLICY,
        YEAR,
        WASTE_CLASS,
        TREATMENT_TYPE,
        DEFAULT_DELAY_COST_USD_PER_T_DAY,
    )
    target_regions = load_target_regions()

    route_frames = []
    summary_rows = []

    objective_runs = [
        ("economic_choice_baseline", "Global", "cost_min", set()),
        ("global_li_max_benchmark", "Global", "max_lithium", set()),
    ]
    objective_runs.extend(
        ("domestic_li_allocation_objective", target_region, "max_target_lithium", target_iso)
        for target_region, target_iso in target_regions.items()
    )

    for objective, target_region, solver_objective, target_iso in objective_runs:
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
        routes["policy_scenario"] = POLICY
        routes["objective"] = objective
        routes["target_region"] = target_region
        route_frames.append(routes)
        summary_rows.append(summarize_routes(routes, objective, target_region, target_iso))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    routes_output = OUTPUT_DIR / "objective_benchmark_routes.csv"
    summary_output = OUTPUT_DIR / "objective_benchmark_summary.csv"
    pd.concat(route_frames, ignore_index=True).to_csv(routes_output, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_output, index=False)
    return routes_output, summary_output


def main():
    routes_output, summary_output = run()
    print(f"Wrote {routes_output}")
    print(f"Wrote {summary_output}")


if __name__ == "__main__":
    main()
