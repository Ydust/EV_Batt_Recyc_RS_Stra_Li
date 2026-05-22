import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix
try:
    import gurobipy as gp
    from gurobipy import GRB
except ImportError:
    gp = None
    GRB = None

import scenario_transport_paths as transport_module
from scenario_transport_paths import (
    BIG_M,
    FORBIDDEN_COST,
    DEFAULT_DELAY_COST_USD_PER_T_DAY,
    apply_policy_constraints,
    load_carbon_unit_cost as base_load_carbon_unit_cost,
    load_distance_matrix,
    load_inputs,
    load_recycling_unit_cost as base_load_recycling_unit_cost,
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
COST_FILE = ROOT / "cost" / "cost_coun_df.csv"
ALL_COUNTRIES_FILE = ROOT / "all_countries.csv"
DEVELOPED_NATION_FILE = ROOT / "developed_nation_list.csv"
TECH_CAPABILITY_FILE = ROOT / "technology_country_capability.csv"
_COST_TABLE_CACHE = None
_CAPABILITY_TABLE_CACHE = None
LP_SOLVER = "highs"
PYROHYDRO_COST_WEIGHTS = {"Pyro": 0.45, "Hydro": 0.55}
PYROHYDRO_GROUP_COST_MULTIPLIER = {
    "developed": 1.00,
    "ev_producer": 1.06,
    "other": 1.15,
}
PYROHYDRO_COUNTRY_COST_MULTIPLIER = {
    "BEL": 0.92,
    "JPN": 0.95,
    "CAN": 0.97,
    "DEU": 1.00,
    "USA": 1.02,
    "CHN": 1.08,
    "KOR": 1.08,
    "IND": 1.18,
}


def configure_pyrohydro(pyro_weight=None, group_multipliers=None, country_multipliers=None):
    if pyro_weight is not None:
        pyro_weight = float(pyro_weight)
        if not 0.0 <= pyro_weight <= 1.0:
            raise ValueError("pyro_weight must be between 0 and 1.")
        PYROHYDRO_COST_WEIGHTS["Pyro"] = pyro_weight
        PYROHYDRO_COST_WEIGHTS["Hydro"] = 1.0 - pyro_weight
    if group_multipliers:
        PYROHYDRO_GROUP_COST_MULTIPLIER.update(
            {key: float(value) for key, value in group_multipliers.items()}
        )
    if country_multipliers:
        PYROHYDRO_COUNTRY_COST_MULTIPLIER.update(
            {key: float(value) for key, value in country_multipliers.items()}
        )


def configure_lp_solver(lp_solver):
    global LP_SOLVER
    if lp_solver not in {"highs", "gurobi"}:
        raise ValueError("lp_solver must be 'highs' or 'gurobi'.")
    if lp_solver == "gurobi" and gp is None:
        raise RuntimeError("gurobipy is not installed in this Python environment.")
    LP_SOLVER = lp_solver


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_years(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_li_content():
    metal = pd.read_csv(METAL_CONTENT_FILE)
    return metal.dropna(subset=["Type"]).set_index("Type")["Li"].astype(float)


def load_recovery_efficiency(recovery_scenario, years, methods):
    recovery = pd.read_csv(RECOVERY_FILE)
    scenario_recovery = recovery[
        (recovery["recovery_efficiency_scenario"] == recovery_scenario)
        & (recovery["Year"].isin(years))
    ].copy()
    values = {
        (int(row["Year"]), row["recycling_m"]): float(row["li_recovery_efficiency"])
        for _, row in scenario_recovery[
            scenario_recovery["recycling_m"].isin(methods)
        ].iterrows()
    }
    if "PyroHydro" in methods:
        for year in years:
            pyro = scenario_recovery[
                (scenario_recovery["Year"] == year)
                & (scenario_recovery["recycling_m"] == "Pyro")
            ]["li_recovery_efficiency"]
            hydro = scenario_recovery[
                (scenario_recovery["Year"] == year)
                & (scenario_recovery["recycling_m"] == "Hydro")
            ]["li_recovery_efficiency"]
            if not pyro.empty and not hydro.empty:
                values[(int(year), "PyroHydro")] = float(
                    PYROHYDRO_COST_WEIGHTS["Pyro"] * pyro.iloc[0]
                    + PYROHYDRO_COST_WEIGHTS["Hydro"] * hydro.iloc[0]
                )
    return values


def load_emission_factor():
    emission = pd.read_csv(EMISSION_FILE).rename(columns={"battery_type": "type"})
    emission["CO2_new"] = pd.to_numeric(emission["CO2_new"], errors="coerce").fillna(0)
    values = {
        (row["type"], row["recycling_m"]): float(row["CO2_new"])
        for _, row in emission.iterrows()
    }
    for battery_type in emission["type"].dropna().unique():
        pyro = values.get((battery_type, "Pyro"))
        hydro = values.get((battery_type, "Hydro"))
        if pyro is not None and hydro is not None:
            values[(battery_type, "PyroHydro")] = float(
                PYROHYDRO_COST_WEIGHTS["Pyro"] * pyro
                + PYROHYDRO_COST_WEIGHTS["Hydro"] * hydro
            )
    return values


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
    destination_countries = country_meta[country_meta["iso3"].isin(capacity.index)].copy()
    transport_cost = distance.reindex(columns=capacity.index, fill_value=BIG_M).copy()
    for method in methods:
        recycling_cost = load_recycling_unit_cost(method, destination_countries)
        carbon_cost = load_carbon_unit_cost(method, destination_countries)
        destination_cost = (
            recycling_cost.add(carbon_cost, fill_value=0.0)
            .reindex(capacity.index)
            .fillna(0.0)
        )
        total_cost = transport_cost.add(destination_cost, axis="columns")
        costs[method] = apply_policy_constraints(
            total_cost,
            country_meta,
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


def load_scale_adjusted_recycling_unit_cost(
    method,
    countries,
    allocated_scrap_by_iso=None,
    require_country_technology_curve=True,
):
    if method == "PyroHydro":
        pyro = load_scale_adjusted_recycling_unit_cost(
            "Pyro",
            countries,
            allocated_scrap_by_iso,
            require_country_technology_curve=require_country_technology_curve,
        )
        hydro = load_scale_adjusted_recycling_unit_cost(
            "Hydro",
            countries,
            allocated_scrap_by_iso,
            require_country_technology_curve=require_country_technology_curve,
        )
        country_groups = load_country_groups_by_iso()
        multipliers = pd.Series(
            {
                iso3: pyrohydro_country_multiplier(
                    iso3, country_groups.get(iso3, "other")
                )
                for iso3 in countries["iso3"].dropna().unique()
            },
            dtype=float,
        )
        return (
            PYROHYDRO_COST_WEIGHTS["Pyro"] * pyro
            + PYROHYDRO_COST_WEIGHTS["Hydro"] * hydro
        ) * multipliers.reindex(pyro.index).fillna(
            PYROHYDRO_GROUP_COST_MULTIPLIER["other"]
        )

    global _COST_TABLE_CACHE
    if _COST_TABLE_CACHE is None:
        _COST_TABLE_CACHE = pd.read_csv(COST_FILE)
    cost = _COST_TABLE_CACHE.copy()
    if method not in cost.columns:
        raise ValueError(f"No recycling cost column found for method: {method}")
    cost = cost[["country", "Recycling_capacity", method]].dropna().copy()
    cost["Recycling_capacity"] = pd.to_numeric(cost["Recycling_capacity"], errors="coerce")
    cost[method] = pd.to_numeric(cost[method], errors="coerce")
    cost = cost.dropna(subset=["Recycling_capacity", method])
    allocated_scrap_by_iso = allocated_scrap_by_iso or {}
    fallback = load_recycling_unit_cost(method, countries)
    fallback_value = float(fallback.median()) if not fallback.empty else 0.0
    values = {}
    for _, country_row in countries.dropna(subset=["iso3", "country"]).iterrows():
        iso3 = country_row["iso3"]
        country = country_row["country"]
        curve = cost[cost["country"] == country].sort_values("Recycling_capacity")
        if curve.empty:
            values[iso3] = (
                np.nan
                if require_country_technology_curve
                else float(fallback.get(iso3, fallback_value))
            )
            continue
        allocated = max(float(allocated_scrap_by_iso.get(iso3, 0.0)), 0.0)
        capacities = curve["Recycling_capacity"].to_numpy(dtype=float)
        unit_costs = curve[method].to_numpy(dtype=float)
        values[iso3] = float(
            np.interp(
                allocated,
                capacities,
                unit_costs,
                left=unit_costs[0],
                right=unit_costs[-1],
            )
        )
    return pd.Series(values, dtype=float)


def pyrohydro_country_multiplier(iso3, country_group):
    return PYROHYDRO_COUNTRY_COST_MULTIPLIER.get(
        iso3,
        PYROHYDRO_GROUP_COST_MULTIPLIER.get(country_group, 1.10),
    )


def load_recycling_unit_cost(method, countries):
    if method != "PyroHydro":
        return base_load_recycling_unit_cost(method, countries)
    pyro = base_load_recycling_unit_cost("Pyro", countries)
    hydro = base_load_recycling_unit_cost("Hydro", countries)
    country_groups = load_country_groups_by_iso()
    multipliers = pd.Series(
        {
            iso3: pyrohydro_country_multiplier(iso3, country_groups.get(iso3, "other"))
            for iso3 in countries["iso3"].dropna().unique()
        },
        dtype=float,
    )
    return (
        PYROHYDRO_COST_WEIGHTS["Pyro"] * pyro
        + PYROHYDRO_COST_WEIGHTS["Hydro"] * hydro
    ) * multipliers.reindex(pyro.index).fillna(
        PYROHYDRO_GROUP_COST_MULTIPLIER["other"]
    )


def load_carbon_unit_cost(method, countries):
    if method != "PyroHydro":
        return base_load_carbon_unit_cost(method, countries)
    pyro = base_load_carbon_unit_cost("Pyro", countries)
    hydro = base_load_carbon_unit_cost("Hydro", countries)
    return PYROHYDRO_COST_WEIGHTS["Pyro"] * pyro + PYROHYDRO_COST_WEIGHTS["Hydro"] * hydro


def build_method_cost_components_with_scale(
    distance,
    capacity,
    country_meta,
    methods,
    policy,
    year,
    waste_class,
    treatment_type,
    delay_cost,
    allocated_scrap_by_method=None,
    require_country_technology_curve=True,
    availability_threshold=None,
    method_cost_multipliers=None,
):
    components = {}
    allocated_scrap_by_method = allocated_scrap_by_method or {}
    method_cost_multipliers = method_cost_multipliers or {}
    destination_countries = country_meta[country_meta["iso3"].isin(capacity.index)].copy()
    transport_cost = distance.reindex(columns=capacity.index, fill_value=BIG_M).copy()
    for method in methods:
        recycling_cost = load_scale_adjusted_recycling_unit_cost(
            method,
            destination_countries,
            allocated_scrap_by_method.get(method, {}),
            require_country_technology_curve=require_country_technology_curve,
        )
        carbon_cost = load_carbon_unit_cost(method, destination_countries)
        destination_recycling = recycling_cost.reindex(capacity.index)
        if method in method_cost_multipliers:
            destination_recycling = destination_recycling * float(method_cost_multipliers[method])
        destination_carbon = carbon_cost.reindex(capacity.index).fillna(0.0)
        pre_policy_cost = transport_cost.add(
            destination_recycling.fillna(0.0).add(destination_carbon, fill_value=0.0),
            axis="columns",
        )
        unavailable_destinations = destination_recycling[
            destination_recycling.isna()
        ].index
        if availability_threshold is not None:
            available_destinations = technology_available_destinations(
                method,
                year,
                destination_countries,
                availability_threshold,
            )
            capability_unavailable = [
                dst for dst in destination_recycling.index if dst not in available_destinations
            ]
            unavailable_destinations = unavailable_destinations.union(capability_unavailable)
        if len(unavailable_destinations) > 0:
            pre_policy_cost.loc[:, unavailable_destinations] = FORBIDDEN_COST
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


def method_costs_from_components(method_components, methods):
    return {method: method_components[method]["total"] for method in methods}


def load_country_groups_by_iso():
    all_countries = pd.read_csv(ALL_COUNTRIES_FILE)
    country_groups = all_countries[["country", "iso3", "producer"]].dropna(subset=["iso3"]).copy()
    country_groups["country_group"] = np.where(
        country_groups["producer"].astype(bool), "ev_producer", "other"
    )
    if DEVELOPED_NATION_FILE.exists():
        developed = pd.read_csv(DEVELOPED_NATION_FILE)
        developed_countries = set(developed["region"].dropna())
        country_groups.loc[
            country_groups["country"].isin(developed_countries), "country_group"
        ] = "developed"
    return country_groups.set_index("iso3")["country_group"].to_dict()


def load_interpolated_capability(year):
    global _CAPABILITY_TABLE_CACHE
    if _CAPABILITY_TABLE_CACHE is None:
        _CAPABILITY_TABLE_CACHE = pd.read_csv(TECH_CAPABILITY_FILE)
    capability = _CAPABILITY_TABLE_CACHE.copy()
    numeric_cols = [
        "year",
        "availability",
        "maturity_score",
        "capability_score",
        "complexity_penalty",
        "policy_bonus",
    ]
    for col in numeric_cols:
        capability[col] = pd.to_numeric(capability[col], errors="coerce")
    rows = []
    for (country_group, method), group in capability.groupby(["country_group", "recycling_m"]):
        group = group.sort_values("year")
        row = {"country_group": country_group, "recycling_m": method, "year": year}
        for col in numeric_cols:
            if col == "year":
                continue
            row[col] = float(np.interp(year, group["year"], group[col]))
        rows.append(row)
    capability = pd.DataFrame(rows)
    hybrid_rows = []
    value_cols = [col for col in numeric_cols if col != "year"]
    for country_group, group in capability.groupby("country_group"):
        by_method = group.set_index("recycling_m")
        if {"Pyro", "Hydro"}.issubset(by_method.index):
            row = {
                "country_group": country_group,
                "recycling_m": "PyroHydro",
                "year": year,
            }
            for col in value_cols:
                pyro_value = float(by_method.loc["Pyro", col])
                hydro_value = float(by_method.loc["Hydro", col])
                row[col] = min(pyro_value, hydro_value) if col == "availability" else (
                    PYROHYDRO_COST_WEIGHTS["Pyro"] * pyro_value
                    + PYROHYDRO_COST_WEIGHTS["Hydro"] * hydro_value
                )
            hybrid_rows.append(row)
    if hybrid_rows:
        capability = pd.concat([capability, pd.DataFrame(hybrid_rows)], ignore_index=True)
    return capability


def technology_available_destinations(method, year, destination_countries, threshold):
    country_groups = load_country_groups_by_iso()
    capability = load_interpolated_capability(year)
    availability = capability[
        capability["recycling_m"] == method
    ].set_index("country_group")["availability"].to_dict()
    available = set()
    for _, row in destination_countries.iterrows():
        group = country_groups.get(row["iso3"], "other")
        if float(availability.get(group, 0.0)) >= float(threshold):
            available.add(row["iso3"])
    return available


def allocated_scrap_by_destination_method(routes, methods):
    real = routes[~routes["is_unprocessed"].astype(bool)].copy()
    allocated = {method: {} for method in methods}
    if real.empty:
        return allocated
    grouped = real.groupby(["technology", "destination_iso3"])["scrap_t"].sum()
    for (method, destination), value in grouped.items():
        if method in allocated:
            allocated[method][destination] = float(value)
    return allocated


def blend_allocated_scrap(previous, current, methods, relaxation):
    if previous is None:
        return current
    blended = {method: {} for method in methods}
    for method in methods:
        destinations = set(previous.get(method, {})) | set(current.get(method, {}))
        for destination in destinations:
            old_value = float(previous.get(method, {}).get(destination, 0.0))
            new_value = float(current.get(method, {}).get(destination, 0.0))
            blended[method][destination] = (
                (1.0 - relaxation) * old_value + relaxation * new_value
            )
    return blended


def route_modeled_cost(routes):
    real = routes[~routes["is_unprocessed"].astype(bool)].copy()
    return float(
        real[["transport_cost", "recycling_cost", "carbon_cost", "policy_cost"]]
        .sum(axis=1)
        .sum()
    )


def solve_joint_dynamic_scale(
    supply,
    capacity,
    distance,
    country_meta,
    methods,
    policy,
    year,
    waste_class,
    treatment_type,
    delay_cost,
    max_iterations=8,
    tolerance=1e-3,
    relaxation=0.5,
    require_country_technology_curve=True,
    availability_threshold=None,
    method_cost_multipliers=None,
):
    allocated = None
    previous_total_cost = None
    routes = None
    components = None
    for iteration in range(max_iterations):
        components = build_method_cost_components_with_scale(
            distance,
            capacity,
            country_meta,
            methods,
            policy,
            year,
            waste_class,
            treatment_type,
            delay_cost,
            allocated,
            require_country_technology_curve=require_country_technology_curve,
            availability_threshold=availability_threshold,
            method_cost_multipliers=method_cost_multipliers,
        )
        routes = solve_joint(
            supply,
            capacity,
            method_costs_from_components(components, methods),
            methods,
            components,
        )
        total_cost = route_modeled_cost(routes)
        routes["scale_iteration"] = iteration + 1
        routes["scale_total_cost"] = total_cost
        if previous_total_cost is not None:
            denominator = max(abs(previous_total_cost), 1.0)
            if abs(total_cost - previous_total_cost) / denominator <= tolerance:
                break
        previous_total_cost = total_cost
        current_allocated = allocated_scrap_by_destination_method(routes, methods)
        allocated = blend_allocated_scrap(allocated, current_allocated, methods, relaxation)
    return routes, components


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

    if LP_SOLVER == "gurobi":
        model = gp.Model("joint_policy_transport_technology")
        model.Params.OutputFlag = 0
        model.Params.Threads = int(os.environ.get("GUROBI_THREADS", "0"))
        x = model.addVars(n_vars, lb=0.0, obj=costs, name="x")
        for i in range(n_supply):
            indices = a_eq.rows[i]
            model.addConstr(
                gp.quicksum(x[j] for j in indices) == float(b_eq[i]),
                name=f"supply_{i}",
            )
        for dst, row_index in destination_index.items():
            indices = a_ub.rows[row_index]
            model.addConstr(
                gp.quicksum(x[j] for j in indices) <= float(capacity_values[dst]),
                name=f"capacity_{dst}",
            )
        for dst, row_index in domestic_priority_index.items():
            indices = a_ub.rows[row_index]
            target = float(domestic_priority_targets[dst])
            model.addConstr(
                gp.quicksum(x[j] for j in indices) >= target,
                name=f"domestic_priority_{dst}",
            )
        model.ModelSense = GRB.MINIMIZE
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Gurobi failed with status {model.Status}")
        solution_values = np.array([x[j].X for j in range(n_vars)], dtype=float)
    else:
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
        solution_values = result.x

    rows = []
    for value, variable, cost in zip(solution_values, variables, costs):
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
