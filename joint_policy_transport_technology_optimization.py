import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

import scenario_transport_paths as transport_module
from scenario_transport_paths import (
    BIG_M,
    FORBIDDEN_COST,
    DEFAULT_DELAY_COST_USD_PER_T_DAY,
    apply_policy_constraints,
    build_total_cost_matrix,
    load_carbon_unit_cost,
    load_distance_matrix,
    load_inputs,
    load_recycling_unit_cost,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "Figure_data" / "joint_policy_technology"
METAL_CONTENT_FILE = ROOT / "cost" / "Metal content.csv"
RECOVERY_FILE = ROOT / "technology_lithium_recovery_scenarios.csv"
EMISSION_FILE = ROOT / "cost" / "recycling_emission_count.csv"
SCENARIO_SCRAP_FILE = (
    ROOT / "Scenario result" / "recycling_rate" / "EV_battery_inuse_scrap_collected_by_scenario.csv"
)
COUNTRY_FILE = ROOT / "nation_list_new.csv"


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_years(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_li_content():
    metal = pd.read_csv(METAL_CONTENT_FILE)
    return metal.dropna(subset=["Type"]).set_index("Type")["Li"].astype(float)


def load_recovery_efficiency(recovery_scenario, years, methods):
    recovery = pd.read_csv(RECOVERY_FILE)
    recovery = recovery[
        (recovery["recovery_efficiency_scenario"] == recovery_scenario)
        & (recovery["Year"].isin(years))
        & (recovery["recycling_m"].isin(methods))
    ].copy()
    return {
        (int(row["Year"]), row["recycling_m"]): float(row["li_recovery_efficiency"])
        for _, row in recovery.iterrows()
    }


def load_emission_factor():
    emission = pd.read_csv(EMISSION_FILE).rename(columns={"battery_type": "type"})
    emission["CO2_new"] = pd.to_numeric(emission["CO2_new"], errors="coerce").fillna(0)
    return {
        (row["type"], row["recycling_m"]): float(row["CO2_new"])
        for _, row in emission.iterrows()
    }


def load_scrap_by_type(collection_scenario):
    scrap = pd.read_csv(SCENARIO_SCRAP_FILE)
    scrap = scrap[scrap["scenario"] == collection_scenario].copy()
    countries = pd.read_csv(COUNTRY_FILE)[["region", "iso3"]].rename(
        columns={"region": "country"}
    )
    scrap = scrap.rename(columns={"region": "country"}).merge(
        countries, on="country", how="left"
    )
    scrap = scrap.dropna(subset=["iso3"])
    return scrap


def make_supply_by_type(scrap_by_type, countries, year, strategy):
    year_scrap = scrap_by_type[scrap_by_type["Year"] == year].copy()
    if strategy == "Strategy 2":
        producer_iso = set(countries.loc[countries["producer"] == True, "iso3"])
        year_scrap = year_scrap[year_scrap["iso3"].isin(producer_iso)].copy()
    elif strategy != "Strategy 3":
        raise ValueError("This joint model currently supports Strategy 2 or Strategy 3.")

    supply = (
        year_scrap.groupby(["iso3", "type"], as_index=False)["scrap"]
        .sum()
        .rename(columns={"iso3": "source_iso3", "scrap": "scrap_t"})
    )
    return supply[supply["scrap_t"] > 0].copy()


def make_capacity(capacity, producer_iso, year):
    year_capacity = capacity[capacity["Year"] == year].copy()
    demand = year_capacity[year_capacity["iso3"].isin(producer_iso)][
        ["iso3", "Mass_v"]
    ].copy()
    demand = demand.groupby("iso3", as_index=True)["Mass_v"].sum().to_frame()
    return demand[demand["Mass_v"] > 0].copy()


def build_method_costs(distance, capacity, country_meta, methods, policy, year, waste_class, treatment_type, delay_cost):
    costs = {}
    for method in methods:
        costs[method] = build_total_cost_matrix(
            distance,
            capacity,
            country_meta,
            method,
            policy,
            year,
            waste_class,
            treatment_type,
            delay_cost,
        )
    return costs


def build_method_cost_components(
    distance,
    capacity,
    country_meta,
    methods,
    policy,
    year,
    waste_class,
    treatment_type,
    delay_cost,
):
    components = {}
    destination_countries = country_meta[country_meta["iso3"].isin(capacity.index)].copy()
    transport_cost = distance.reindex(columns=capacity.index, fill_value=BIG_M).copy()
    for method in methods:
        recycling_cost = load_recycling_unit_cost(method, destination_countries)
        carbon_cost = load_carbon_unit_cost(method, destination_countries)
        destination_recycling = recycling_cost.reindex(capacity.index).fillna(0.0)
        destination_carbon = carbon_cost.reindex(capacity.index).fillna(0.0)
        pre_policy_cost = transport_cost.add(
            destination_recycling.add(destination_carbon, fill_value=0.0),
            axis="columns",
        )
        total_cost = apply_policy_constraints(
            pre_policy_cost,
            country_meta,
            policy,
            year,
            waste_class,
            treatment_type,
            delay_cost,
        )
        policy_cost = total_cost.subtract(pre_policy_cost, fill_value=0.0)
        policy_cost = policy_cost.where(total_cost < FORBIDDEN_COST, FORBIDDEN_COST)
        components[method] = {
            "total": total_cost,
            "transport": transport_cost,
            "recycling": destination_recycling,
            "carbon": destination_carbon,
            "policy": policy_cost,
        }
    return components


def get_component_unit_cost(method_components, method, component, src, dst):
    values = method_components[method][component]
    if isinstance(values, pd.Series):
        return float(values.get(dst, 0.0))
    if src not in values.index or dst not in values.columns:
        return 0.0
    value = float(values.loc[src, dst])
    return 0.0 if value >= FORBIDDEN_COST else value


def solve_joint(supply, capacity, method_costs, methods, method_components=None):
    supply = supply.copy().reset_index(drop=True)
    destinations = capacity.index.tolist()
    capacity_values = capacity["Mass_v"].to_dict()
    domestic_supply = supply.groupby("source_iso3")["scrap_t"].sum().to_dict()

    variables = []
    costs = []
    supply_row = {}

    for i, row in supply.iterrows():
        src = row["source_iso3"]
        src_type = row["type"]
        supply_row[(src, src_type)] = i
        for dst in destinations:
            for method in methods:
                cost_matrix = method_costs[method]
                if src not in cost_matrix.index or dst not in cost_matrix.columns:
                    continue
                cost = float(cost_matrix.loc[src, dst])
                if cost >= FORBIDDEN_COST:
                    continue
                variables.append((src, src_type, dst, method, False))
                costs.append(cost)
        variables.append((src, src_type, "Virtual_Demand", "Unprocessed", True))
        costs.append(BIG_M)

    n_vars = len(variables)
    n_supply = len(supply)
    n_dest = len(destinations)
    domestic_route_available = {
        src
        for src, _, dst, _, is_virtual in variables
        if not is_virtual and src == dst
    }
    domestic_priority_targets = {
        dst: min(float(domestic_supply.get(dst, 0.0)), float(capacity_values[dst]))
        for dst in destinations
        if dst in domestic_route_available
        and min(float(domestic_supply.get(dst, 0.0)), float(capacity_values[dst])) > 1e-9
    }

    a_eq = lil_matrix((n_supply, n_vars))
    b_eq = np.zeros(n_supply)
    for i, row in supply.iterrows():
        b_eq[i] = float(row["scrap_t"])

    n_domestic_priority = len(domestic_priority_targets)
    a_ub = lil_matrix((n_dest + n_domestic_priority, n_vars))
    b_ub = np.array(
        [capacity_values[dst] for dst in destinations]
        + [-target for target in domestic_priority_targets.values()],
        dtype=float,
    )
    destination_index = {dst: i for i, dst in enumerate(destinations)}
    domestic_priority_index = {
        dst: n_dest + i for i, dst in enumerate(domestic_priority_targets)
    }

    for j, (src, src_type, dst, method, is_virtual) in enumerate(variables):
        a_eq[supply_row[(src, src_type)], j] = 1
        if not is_virtual:
            a_ub[destination_index[dst], j] = 1
            if src == dst and dst in domestic_priority_index:
                a_ub[domestic_priority_index[dst], j] = -1

    result = linprog(
        np.array(costs, dtype=float),
        A_ub=a_ub.tocsr(),
        b_ub=b_ub,
        A_eq=a_eq.tocsr(),
        b_eq=b_eq,
        bounds=(0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)

    rows = []
    for value, variable, cost in zip(result.x, variables, costs):
        if value <= 1e-9:
            continue
        src, src_type, dst, method, is_virtual = variable
        transport_unit_cost = 0.0
        recycling_unit_cost = 0.0
        carbon_unit_cost = 0.0
        policy_unit_cost = 0.0
        if method_components is not None and not is_virtual:
            transport_unit_cost = get_component_unit_cost(
                method_components, method, "transport", src, dst
            )
            recycling_unit_cost = get_component_unit_cost(
                method_components, method, "recycling", src, dst
            )
            carbon_unit_cost = get_component_unit_cost(
                method_components, method, "carbon", src, dst
            )
            policy_unit_cost = get_component_unit_cost(
                method_components, method, "policy", src, dst
            )
        rows.append(
            {
                "source_iso3": src,
                "battery_type": src_type,
                "destination_iso3": dst,
                "technology": method,
                "scrap_t": float(value),
                "unit_cost": float(cost),
                "transport_unit_cost": transport_unit_cost,
                "recycling_unit_cost": recycling_unit_cost,
                "carbon_unit_cost": carbon_unit_cost,
                "policy_unit_cost": policy_unit_cost,
                "transport_cost": float(value) * transport_unit_cost,
                "recycling_cost": float(value) * recycling_unit_cost,
                "carbon_cost": float(value) * carbon_unit_cost,
                "policy_cost": float(value) * policy_unit_cost,
                "is_unprocessed": bool(is_virtual),
                "domestic_priority_reserved": (
                    (not is_virtual)
                    and src == dst
                    and dst in domestic_priority_targets
                ),
            }
        )
    return pd.DataFrame(rows)


def solve_joint_domestic_priority(supply, capacity, method_costs, methods, method_components=None):
    return solve_joint(supply, capacity, method_costs, methods, method_components)


def add_lithium_outputs(routes, year, recovery_efficiency, li_content, emission_factor):
    routes = routes.copy()
    routes["li_content"] = routes["battery_type"].map(li_content).fillna(0.0)
    routes["contained_lithium_t"] = routes["scrap_t"] * routes["li_content"]
    routes["li_recovery_efficiency"] = routes["technology"].map(
        lambda tech: recovery_efficiency.get((year, tech), 0.0)
    )
    routes["recovered_lithium_t"] = (
        routes["contained_lithium_t"] * routes["li_recovery_efficiency"]
    )
    routes.loc[routes["is_unprocessed"], "recovered_lithium_t"] = 0.0
    routes["battery_embedded_secondary_li_t"] = routes["recovered_lithium_t"]
    routes["CO2_new_g_per_kg"] = routes.apply(
        lambda row: emission_factor.get((row["battery_type"], row["technology"]), 0.0),
        axis=1,
    )
    routes["recycling_CO2_t"] = routes["scrap_t"] * routes["CO2_new_g_per_kg"] / 1000.0
    routes.loc[routes["is_unprocessed"], "recycling_CO2_t"] = 0.0
    return routes


def summarize(routes, year, policy, strategy):
    real = routes[~routes["is_unprocessed"]].copy()
    return {
        "year": year,
        "policy_scenario": policy,
        "strategy": strategy,
        "processed_scrap_t": float(real["scrap_t"].sum()),
        "unprocessed_scrap_t": float(routes.loc[routes["is_unprocessed"], "scrap_t"].sum()),
        "contained_lithium_t": float(real["contained_lithium_t"].sum()),
        "recovered_lithium_t": float(real["recovered_lithium_t"].sum()),
        "recycling_CO2_t": float(real["recycling_CO2_t"].sum()),
        "total_objective_cost": float((routes["scrap_t"] * routes["unit_cost"]).sum()),
        "cross_border_scrap_t": float(
            real.loc[real["source_iso3"] != real["destination_iso3"], "scrap_t"].sum()
        ),
    }


def run(
    collection_scenario,
    recovery_scenario,
    policies,
    years,
    methods,
    strategy,
    waste_class,
    treatment_type,
    delay_cost,
    policy_file,
):
    transport_module.POLICY_FILE = Path(policy_file)
    _, capacity, producer_iso, country_meta = load_inputs(collection_scenario)
    scrap_by_type = load_scrap_by_type(collection_scenario)
    countries = pd.read_csv(ROOT / "all_countries.csv")
    distance = load_distance_matrix()
    li_content = load_li_content()
    recovery = load_recovery_efficiency(recovery_scenario, years, methods)
    emission = load_emission_factor()

    route_frames = []
    summary_rows = []
    for year in years:
        supply = make_supply_by_type(scrap_by_type, countries, year, strategy)
        destination_capacity = make_capacity(capacity, producer_iso, year)
        for policy in policies:
            method_costs = build_method_costs(
                distance,
                destination_capacity,
                country_meta,
                methods,
                policy,
                year,
                waste_class,
                treatment_type,
                delay_cost,
            )
            method_components = build_method_cost_components(
                distance,
                destination_capacity,
                country_meta,
                methods,
                policy,
                year,
                waste_class,
                treatment_type,
                delay_cost,
            )
            routes = solve_joint_domestic_priority(
                supply, destination_capacity, method_costs, methods, method_components
            )
            routes = add_lithium_outputs(routes, year, recovery, li_content, emission)
            routes["year"] = year
            routes["policy_scenario"] = policy
            routes["strategy"] = strategy
            route_frames.append(routes)
            summary_rows.append(summarize(routes, year, policy, strategy))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    route_output = OUTPUT_DIR / "joint_policy_transport_technology_routes.csv"
    summary_output = OUTPUT_DIR / "joint_policy_transport_technology_summary.csv"
    pd.concat(route_frames, ignore_index=True).to_csv(route_output, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_output, index=False)
    return route_output, summary_output


def main():
    parser = argparse.ArgumentParser(
        description="Jointly optimize policy-constrained transport routes and economically selected recycling technology."
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
    parser.add_argument("--years", default="2030")
    parser.add_argument("--methods", default="Direct,Hydro,Pyro")
    parser.add_argument("--strategy", default="Strategy 3")
    parser.add_argument("--waste-class", default="hazardous")
    parser.add_argument("--treatment-type", default="recovery")
    parser.add_argument("--delay-cost", type=float, default=DEFAULT_DELAY_COST_USD_PER_T_DAY)
    parser.add_argument(
        "--policy-file",
        default=str(ROOT / "waste_trade_policy_constraints_with_critical_routes.csv"),
    )
    args = parser.parse_args()

    route_output, summary_output = run(
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
    print(f"Wrote {route_output}")
    print(f"Wrote {summary_output}")


if __name__ == "__main__":
    main()
