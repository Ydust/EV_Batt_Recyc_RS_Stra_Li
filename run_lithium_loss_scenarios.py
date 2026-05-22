import argparse
import itertools
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix, vstack

import scenario_transport_paths as transport_module
from joint_policy_transport_technology_optimization import (
    OUTPUT_DIR,
    add_lithium_outputs,
    build_method_cost_components,
    build_method_costs,
    get_component_unit_cost,
    load_emission_factor,
    load_li_content,
    load_recovery_efficiency,
    load_scrap_by_type,
    make_capacity,
    make_supply_by_type,
)
from scenario_transport_paths import (
    BIG_M,
    DEFAULT_DELAY_COST_USD_PER_T_DAY,
    FORBIDDEN_COST,
    load_distance_matrix,
    load_inputs,
)


ROOT = Path(__file__).resolve().parent
LITHIUM_PRICE_FILE = ROOT / "lithium_price_scenario.csv"
POLICY_FILE = ROOT / "Figure_data" / "joint_policy_technology" / "waste_trade_policy_constraints_reference_relaxed.csv"
SCENARIO_OUTPUT_DIR = OUTPUT_DIR / "lithium_loss_scenarios"


SCENARIOS = {
    "baseline": {
        "description": "Cost-minimizing joint transport and technology choice.",
    },
    "lithium_aware_high_price": {
        "description": "Internalize recovered lithium value with a high lithium shadow price.",
        "lithium_aware": True,
        "lithium_price_multiplier": 10.0,
    },
    "high_direct_maturity": {
        "description": "Lower Direct recycling cost and improve Direct Li recovery.",
        "direct_cost_multiplier": 0.65,
        "direct_recovery_floor": 0.95,
    },
    "high_recovery_efficiency": {
        "description": "Improve technology-specific Li recovery without changing costs.",
        "recovery_floor_by_method": {
            "Direct": 0.95,
            "Hydro": 0.95,
            "Pyro": 0.35,
        },
    },
    "capacity_expansion": {
        "description": "Expand capacity in key destination countries.",
        "capacity_multiplier": 1.25,
    },
    "policy_relaxation": {
        "description": "Use open-policy route costs for all policy scenarios.",
        "policy_override": "open_policy",
    },
    "combined_mitigation": {
        "description": (
            "Combine Direct maturity, high Li recovery, high Li value, and key "
            "destination capacity expansion."
        ),
        "lithium_aware": True,
        "lithium_price_multiplier": 10.0,
        "direct_cost_multiplier": 0.65,
        "recovery_floor_by_method": {
            "Direct": 0.97,
            "Hydro": 0.95,
            "Pyro": 0.35,
        },
        "capacity_multiplier": 1.25,
    },
    "black_mass": {
        "description": (
            "#1 sensitivity: allow domestic shredding to black mass before "
            "export; black mass is non-hazardous and exempt from route-access "
            "policy."
        ),
        "black_mass": True,
    },
    "max_lithium": {
        "description": (
            "Upper-bound case that maximizes recovered lithium under advanced "
            "technology and key destination capacity expansion."
        ),
        "objective": "max_lithium",
        "recovery_floor_by_method": {
            "Direct": 0.97,
            "Hydro": 0.95,
            "Pyro": 0.35,
        },
        "capacity_multiplier": 1.25,
    },
}

# --- Full-factorial mitigation factors -------------------------------------
# Six independent mitigation factors. Each factor contributes a disjoint set of
# config keys, so a multi-factor scenario is just the union of these fragments.
MITIGATION_FACTORS = {
    "price": {
        "label": "Lithium value internalization",
        "config": {"lithium_aware": True, "lithium_price_multiplier": 10.0},
    },
    "direct": {
        "label": "Direct recycling maturity",
        "config": {"direct_cost_multiplier": 0.65, "direct_recovery_floor": 0.95},
    },
    "recovery": {
        "label": "High recovery efficiency",
        "config": {
            "recovery_floor_by_method": {"Direct": 0.95, "Hydro": 0.95, "Pyro": 0.35}
        },
    },
    "capacity": {
        "label": "Key-destination capacity expansion",
        "config": {"capacity_multiplier": 1.25},
    },
    "policy": {
        "label": "Route-access policy relaxation",
        "config": {"policy_override": "open_policy"},
    },
    "maxli": {
        "label": "Lithium-maximizing objective",
        "config": {"objective": "max_lithium"},
    },
}

# Fixed factor order used to build deterministic scenario names ("a+b+c").
FACTOR_ORDER = ["price", "direct", "recovery", "capacity", "policy", "maxli"]


def build_factorial_scenarios():
    """Build baseline + every non-empty combination of the 6 mitigation factors.

    Total = 1 baseline + sum_{k=1..6} C(6, k) = 1 + 63 = 64 scenarios.
    """
    scenarios = {
        "baseline": {
            "description": "Cost-minimizing joint transport and technology choice.",
        }
    }
    for size in range(1, len(FACTOR_ORDER) + 1):
        for combo in itertools.combinations(FACTOR_ORDER, size):
            name = "+".join(combo)
            config = {}
            for factor in combo:
                config.update(MITIGATION_FACTORS[factor]["config"])
            labels = ", ".join(MITIGATION_FACTORS[factor]["label"] for factor in combo)
            config["description"] = f"[{size}-factor] {labels}"
            scenarios[name] = config
    return scenarios


PARETO_YEAR = 2050
PARETO_POLICIES = ["reference_policy", "strict_policy", "critical_route_policy"]
PARETO_STEPS = np.linspace(0.0, 1.0, 6)
PARETO_CONFIG = {
    "description": (
        "Cost-recovery frontier under advanced technology and key destination "
        "capacity expansion."
    ),
    "recovery_floor_by_method": {
        "Direct": 0.97,
        "Hydro": 0.95,
        "Pyro": 0.35,
    },
    "capacity_multiplier": 1.25,
}


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_years(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def load_lithium_price(price_scenario):
    price = pd.read_csv(LITHIUM_PRICE_FILE)
    price = price[price["price_scenario"] == price_scenario].copy()
    if price.empty:
        raise ValueError(f"No lithium price scenario found: {price_scenario}")
    return {
        int(row["Year"]): float(row["lithium_price_usd_per_t"])
        for _, row in price.iterrows()
    }


def scale_lithium_price(price, multiplier):
    if multiplier is None:
        return price
    return {year: value * float(multiplier) for year, value in price.items()}


def apply_direct_cost_multiplier(method_costs, method_components, multiplier):
    if multiplier is None or "Direct" not in method_costs:
        return
    method_costs["Direct"] = method_costs["Direct"] * multiplier
    if "Direct" not in method_components:
        return
    for component in ["total", "recycling"]:
        if component in method_components["Direct"]:
            method_components["Direct"][component] = (
                method_components["Direct"][component] * multiplier
            )


def apply_direct_recovery_floor(recovery, years, floor):
    if floor is None:
        return recovery
    adjusted = dict(recovery)
    for year in years:
        key = (year, "Direct")
        adjusted[key] = max(float(adjusted.get(key, 0.0)), float(floor))
    return adjusted


def apply_recovery_floor_by_method(recovery, years, floors):
    if not floors:
        return recovery
    adjusted = dict(recovery)
    for year in years:
        for method, floor in floors.items():
            key = (year, method)
            adjusted[key] = max(float(adjusted.get(key, 0.0)), float(floor))
    return adjusted


def expand_key_capacity(capacity, countries, multiplier):
    if multiplier is None:
        return capacity
    expanded = capacity.copy()
    existing = [iso for iso in countries if iso in expanded.index]
    expanded.loc[existing, "Mass_v"] = expanded.loc[existing, "Mass_v"] * multiplier
    return expanded


def lithium_value_credit_per_kg_battery(src_type, method, year, li_content, recovery, price):
    lithium_t_per_t_battery = float(li_content.get(src_type, 0.0)) * float(
        recovery.get((year, method), 0.0)
    )
    usd_per_t_battery = lithium_t_per_t_battery * float(price.get(year, 0.0))
    return usd_per_t_battery / 1000.0


def lithium_yield_t_per_t_battery(src_type, method, year, li_content, recovery):
    return float(li_content.get(src_type, 0.0)) * float(recovery.get((year, method), 0.0))


def route_modeled_cost(routes):
    real = routes[~routes["is_unprocessed"].astype(bool)].copy()
    return float(
        real[
            ["transport_cost", "recycling_cost", "carbon_cost", "policy_cost"]
        ].sum(axis=1).sum()
    )


# --- #1 Black-mass pre-processing factor ------------------------------------
# Spent batteries can be shredded domestically into "black mass" before export.
# Black mass is treated as non-hazardous and is therefore fully exempt from
# route-access policy. It is modelled as extra pseudo-methods (e.g. "Hydro_BM").
#
# Literature-informed default assumptions (configurable):
#   BLACK_MASS_YIELD -- black mass is roughly 20-30% of total battery pack mass
#     (MDPI Batteries 2023, 9(10):514, doi:10.3390/batteries9100514;
#      J. Power Sources Advances, ScienceDirect S2949823624000278). 0.25 used.
#   BLACK_MASS_SHRED_COST_USD_PER_T -- mechanical pre-treatment (discharge,
#     dismantle, shred, sieve) operating cost, order ~200-500 USD per tonne of
#     battery input; 400 used as midpoint. Model cost matrices are in USD/kg
#     (USD/t divided by 1000, matching the policy-penalty convention).
BLACK_MASS_YIELD = 0.25
BLACK_MASS_SHRED_COST_USD_PER_T = 400.0
BLACK_MASS_SUFFIX = "_BM"


def build_black_mass_costs(
    distance, capacity, country_meta, methods, year, waste_class, treatment_type, delay_cost
):
    """Cost matrices/components for black-mass pseudo-methods (e.g. "Hydro_BM").

    A black-mass route shreds the battery in the source country, then ships the
    concentrated black mass: transport is scaled by BLACK_MASS_YIELD and a
    shredding cost is added at the source. No route-access policy is applied.
    """
    components = build_method_cost_components(
        distance,
        capacity,
        country_meta,
        methods,
        "open_policy",
        year,
        waste_class,
        treatment_type,
        delay_cost,
    )
    shred = BLACK_MASS_SHRED_COST_USD_PER_T / 1000.0
    bm_costs = {}
    bm_components = {}
    for method in methods:
        comp = components[method]
        transport = comp["transport"]
        # Discount real distances only; keep unreachable cells (BIG_M fill) as
        # BIG_M so the yield discount cannot turn them into viable cheap routes.
        bm_transport = transport.where(
            transport >= BIG_M, transport * BLACK_MASS_YIELD
        )
        recycling = comp["recycling"]
        carbon = comp["carbon"]
        total = (
            bm_transport.add(recycling.add(carbon, fill_value=0.0), axis="columns")
            + shred
        )
        bm_name = method + BLACK_MASS_SUFFIX
        bm_costs[bm_name] = total
        bm_components[bm_name] = {
            "total": total,
            "transport": bm_transport,
            "recycling": recycling,
            "carbon": carbon,
            "policy": bm_transport * 0.0,
        }
    return bm_costs, bm_components


def extend_recovery_emission_with_black_mass(recovery, emission, methods, years):
    """Alias each method's Li recovery and CO2 factor onto its "_BM" twin."""
    recovery = dict(recovery)
    for year in years:
        for method in methods:
            recovery[(year, method + BLACK_MASS_SUFFIX)] = recovery.get(
                (year, method), 0.0
            )
    emission = dict(emission)
    for (battery_type, technology), value in list(emission.items()):
        if technology in methods:
            emission[(battery_type, technology + BLACK_MASS_SUFFIX)] = value
    return recovery, emission


def solve_joint_scenario(
    supply,
    capacity,
    method_costs,
    methods,
    method_components,
    year,
    li_content,
    recovery,
    lithium_price,
    lithium_aware=False,
    objective="cost_min",
    min_recovered_lithium_t=None,
    min_target_recovered_lithium_t=None,
    target_destinations=None,
    solver_methods=None,
):
    supply = supply.copy().reset_index(drop=True)
    destinations = capacity.index.tolist()
    capacity_values = capacity["Mass_v"].to_dict()
    domestic_supply = supply.groupby("source_iso3")["scrap_t"].sum().to_dict()

    target_destinations = set(target_destinations or [])
    variables = []
    actual_costs = []
    objective_costs = []
    lithium_yields = []
    target_lithium_yields = []
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
                objective_cost = cost
                lithium_yield = lithium_yield_t_per_t_battery(
                    src_type, method, year, li_content, recovery
                )
                if lithium_aware:
                    objective_cost -= lithium_value_credit_per_kg_battery(
                        src_type, method, year, li_content, recovery, lithium_price
                    )
                variables.append((src, src_type, dst, method, False))
                actual_costs.append(cost)
                objective_costs.append(objective_cost)
                lithium_yields.append(lithium_yield)
                target_lithium_yields.append(
                    lithium_yield if dst in target_destinations else 0.0
                )
        variables.append((src, src_type, "Virtual_Demand", "Unprocessed", True))
        actual_costs.append(BIG_M)
        objective_costs.append(BIG_M)
        lithium_yields.append(0.0)
        target_lithium_yields.append(0.0)

    n_vars = len(variables)
    n_supply = len(supply)
    n_dest = len(destinations)
    domestic_route_available = {
        src for src, _, dst, _, is_virtual in variables if not is_virtual and src == dst
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

    solver_methods = solver_methods or ["highs", "highs-ds", "highs-ipm"]
    last_solver_method = None

    def solve_lp_gurobi(objective_values, a_ub_solve, b_ub_solve):
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            return SimpleNamespace(
                success=False,
                message="gurobipy is not installed",
                x=None,
            )

        c = np.array(objective_values, dtype=float)
        try:
            model = gp.Model("secondary_lithium_lp")
            model.Params.OutputFlag = 0
            model.Params.Method = -1
            model.Params.FeasibilityTol = 1e-7
            model.Params.OptimalityTol = 1e-7
            x = model.addMVar(shape=len(c), lb=0.0, obj=c, name="x")
            model.addMConstr(a_eq.tocsr(), x, "=", b_eq, name="supply")
            model.addMConstr(a_ub_solve.tocsr(), x, "<", b_ub_solve, name="capacity")
            model.ModelSense = GRB.MINIMIZE
            model.optimize()
            if model.Status == GRB.OPTIMAL:
                return SimpleNamespace(
                    success=True,
                    message="Optimization terminated successfully.",
                    x=np.array(x.X, dtype=float),
                )
            status_name = {
                GRB.INFEASIBLE: "infeasible",
                GRB.INF_OR_UNBD: "infeasible_or_unbounded",
                GRB.UNBOUNDED: "unbounded",
                GRB.NUMERIC: "numeric",
                GRB.TIME_LIMIT: "time_limit",
            }.get(model.Status, f"status_{model.Status}")
            return SimpleNamespace(success=False, message=f"Gurobi {status_name}", x=None)
        except Exception as exc:
            return SimpleNamespace(success=False, message=f"Gurobi exception: {exc}", x=None)

    def solve_lp(objective_values, recovered_target=None, target_recovered_target=None):
        nonlocal last_solver_method
        a_ub_solve = a_ub.copy()
        b_ub_solve = b_ub.copy()
        if recovered_target is not None and recovered_target > 0:
            target_row = lil_matrix((1, n_vars))
            for col, lithium_yield in enumerate(lithium_yields):
                target_row[0, col] = -lithium_yield
            a_ub_solve = vstack([a_ub_solve.tocsr(), target_row.tocsr()]).tolil()
            b_ub_solve = np.append(b_ub_solve, -float(recovered_target))
        if target_recovered_target is not None and target_recovered_target > 0:
            target_row = lil_matrix((1, n_vars))
            for col, target_yield in enumerate(target_lithium_yields):
                target_row[0, col] = -target_yield
            a_ub_solve = vstack([a_ub_solve.tocsr(), target_row.tocsr()]).tolil()
            b_ub_solve = np.append(b_ub_solve, -float(target_recovered_target))
        messages = []
        for method in solver_methods:
            if method.lower() == "gurobi":
                result = solve_lp_gurobi(objective_values, a_ub_solve, b_ub_solve)
            else:
                result = linprog(
                    np.array(objective_values, dtype=float),
                    A_ub=a_ub_solve.tocsr(),
                    b_ub=b_ub_solve,
                    A_eq=a_eq.tocsr(),
                    b_eq=b_eq,
                    bounds=(0, None),
                    method=method,
                )
            messages.append(f"{method}: {result.message}")
            if result.success:
                last_solver_method = method
                return result
        result.solver_messages = messages
        return result

    if objective == "max_lithium":
        result = solve_lp([-value for value in lithium_yields])
        if not result.success:
            raise RuntimeError(" | ".join(getattr(result, "solver_messages", [result.message])))
        max_recovered_lithium = float(np.dot(result.x, np.array(lithium_yields)))
        result = solve_lp(actual_costs, max_recovered_lithium * (1.0 - 1e-6))
    elif objective == "max_target_lithium":
        result = solve_lp([-value for value in target_lithium_yields])
        if not result.success:
            raise RuntimeError(" | ".join(getattr(result, "solver_messages", [result.message])))
        max_target_lithium = float(
            np.dot(result.x, np.array(target_lithium_yields))
        )
        result = solve_lp(
            [-value for value in lithium_yields],
            target_recovered_target=max_target_lithium * (1.0 - 1e-6),
        )
        if not result.success:
            raise RuntimeError(" | ".join(getattr(result, "solver_messages", [result.message])))
        max_global_lithium_at_target = float(
            np.dot(result.x, np.array(lithium_yields))
        )
        result = solve_lp(
            actual_costs,
            recovered_target=max_global_lithium_at_target * (1.0 - 1e-6),
            target_recovered_target=max_target_lithium * (1.0 - 1e-6),
        )
    else:
        result = solve_lp(
            objective_costs,
            min_recovered_lithium_t,
            min_target_recovered_lithium_t,
        )
    if not result.success:
        raise RuntimeError(" | ".join(getattr(result, "solver_messages", [result.message])))

    rows = []
    for value, variable, actual_cost, objective_cost, lithium_yield in zip(
        result.x, variables, actual_costs, objective_costs, lithium_yields
    ):
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
                "unit_cost": float(actual_cost),
                "objective_unit_cost": float(objective_cost),
                "lithium_yield_t_per_t_battery": float(lithium_yield),
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
                "solver_method": last_solver_method,
            }
        )
    return pd.DataFrame(rows)


def potential_lithium_t(supply, li_content):
    supply = supply.copy()
    supply["li_content"] = supply["type"].map(li_content).fillna(0.0)
    return float((supply["scrap_t"] * supply["li_content"]).sum())


def summarize_routes(routes, potential_li):
    real = routes[~routes["is_unprocessed"].astype(bool)].copy()
    unprocessed = routes[routes["is_unprocessed"].astype(bool)].copy()
    contained_processed = float(real["contained_lithium_t"].sum())
    recovered = float(real["recovered_lithium_t"].sum())
    unprocessed_li = float(unprocessed["contained_lithium_t"].sum())
    technology_loss = contained_processed - recovered
    total_loss = potential_li - recovered
    return {
        "potential_lithium_t": potential_li,
        "processed_contained_lithium_t": contained_processed,
        "recovered_lithium_t": recovered,
        "unprocessed_lithium_t": unprocessed_li,
        "technology_recovery_loss_t": technology_loss,
        "total_lithium_loss_t": total_loss,
        "processed_scrap_t": float(real["scrap_t"].sum()),
        "unprocessed_scrap_t": float(unprocessed["scrap_t"].sum()),
    }


def route_access_loss(routes, policy, reference_policy="route_access_open"):
    recovered = (
        routes.groupby("policy_scenario")["recovered_lithium_t"].sum().to_dict()
    )
    if policy not in recovered or reference_policy not in recovered:
        return 0.0
    return max(0.0, float(recovered[reference_policy]) - float(recovered[policy]))


def route_access_displaced_lithium(routes, policy, reference_policy="route_access_open"):
    subset = routes[routes["policy_scenario"].isin([policy, reference_policy])].copy()
    if subset.empty or reference_policy not in set(subset["policy_scenario"]):
        return 0.0
    table = (
        subset.groupby(
            ["policy_scenario", "source_iso3", "destination_iso3"], as_index=False
        )["battery_embedded_secondary_li_t"]
        .sum()
        .pivot_table(
            index=["source_iso3", "destination_iso3"],
            columns="policy_scenario",
            values="battery_embedded_secondary_li_t",
            fill_value=0.0,
        )
    )
    if policy not in table.columns:
        return 0.0
    delta = table[policy] - table[reference_policy]
    return float((-delta[delta < 0]).sum())


def run_pareto_frontier(
    capacity,
    producer_iso,
    country_meta,
    scrap_by_type,
    countries,
    distance,
    li_content,
    base_recovery,
    emission,
    lithium_price,
    policies,
    years,
    methods,
    strategy,
    waste_class,
    treatment_type,
    delay_cost,
    capacity_expansion_countries,
):
    columns = [
        "year",
        "policy_scenario",
        "frontier_step",
        "target_share_of_cost_to_max_li_gap",
        "target_recovered_lithium_t",
        "recovered_lithium_t",
        "cost_min_recovered_lithium_t",
        "max_lithium_t",
        "processed_scrap_t",
        "unprocessed_scrap_t",
        "route_modeled_cost",
        "transport_cost",
        "recycling_cost",
        "carbon_cost",
        "policy_cost",
        "scenario_description",
    ]
    if PARETO_YEAR not in years:
        return pd.DataFrame(columns=columns)

    recovery = apply_recovery_floor_by_method(
        base_recovery, years, PARETO_CONFIG.get("recovery_floor_by_method")
    )
    supply = make_supply_by_type(scrap_by_type, countries, PARETO_YEAR, strategy)
    destination_capacity = make_capacity(capacity, producer_iso, PARETO_YEAR)
    destination_capacity = expand_key_capacity(
        destination_capacity,
        capacity_expansion_countries,
        PARETO_CONFIG.get("capacity_multiplier"),
    )

    rows = []
    selected_policies = [policy for policy in PARETO_POLICIES if policy in policies]
    for policy in selected_policies:
        method_costs = build_method_costs(
            distance,
            destination_capacity,
            country_meta,
            methods,
            policy,
            PARETO_YEAR,
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
            PARETO_YEAR,
            waste_class,
            treatment_type,
            delay_cost,
        )
        cost_min_routes = solve_joint_scenario(
            supply,
            destination_capacity,
            method_costs,
            methods,
            method_components,
            PARETO_YEAR,
            li_content,
            recovery,
            lithium_price,
            objective="cost_min",
        )
        cost_min_routes = add_lithium_outputs(
            cost_min_routes, PARETO_YEAR, recovery, li_content, emission
        )
        max_li_routes = solve_joint_scenario(
            supply,
            destination_capacity,
            method_costs,
            methods,
            method_components,
            PARETO_YEAR,
            li_content,
            recovery,
            lithium_price,
            objective="max_lithium",
        )
        max_li_routes = add_lithium_outputs(
            max_li_routes, PARETO_YEAR, recovery, li_content, emission
        )
        cost_min_recovered = float(cost_min_routes["recovered_lithium_t"].sum())
        max_recovered = float(max_li_routes["recovered_lithium_t"].sum())
        gain = max(0.0, max_recovered - cost_min_recovered)

        for step_index, share in enumerate(PARETO_STEPS):
            target = cost_min_recovered + gain * float(share)
            routes = solve_joint_scenario(
                supply,
                destination_capacity,
                method_costs,
                methods,
                method_components,
                PARETO_YEAR,
                li_content,
                recovery,
                lithium_price,
                objective="cost_min",
                min_recovered_lithium_t=target,
            )
            routes = add_lithium_outputs(routes, PARETO_YEAR, recovery, li_content, emission)
            real = routes[~routes["is_unprocessed"].astype(bool)].copy()
            unprocessed = routes[routes["is_unprocessed"].astype(bool)].copy()
            rows.append(
                {
                    "year": PARETO_YEAR,
                    "policy_scenario": policy,
                    "frontier_step": step_index,
                    "target_share_of_cost_to_max_li_gap": float(share),
                    "target_recovered_lithium_t": target,
                    "recovered_lithium_t": float(routes["recovered_lithium_t"].sum()),
                    "cost_min_recovered_lithium_t": cost_min_recovered,
                    "max_lithium_t": max_recovered,
                    "processed_scrap_t": float(real["scrap_t"].sum()),
                    "unprocessed_scrap_t": float(unprocessed["scrap_t"].sum()),
                    "route_modeled_cost": route_modeled_cost(routes),
                    "transport_cost": float(real["transport_cost"].sum()),
                    "recycling_cost": float(real["recycling_cost"].sum()),
                    "carbon_cost": float(real["carbon_cost"].sum()),
                    "policy_cost": float(real["policy_cost"].sum()),
                    "scenario_description": PARETO_CONFIG["description"],
                }
            )

    return pd.DataFrame(rows, columns=columns)


# Per-worker model inputs, populated once by _init_worker in each process.
_WORKER_CTX = {}


def _init_worker(params):
    """Load all model inputs once per worker process (Windows spawn-safe).

    Each worker reloads inputs itself rather than receiving pickled DataFrames;
    load time is negligible next to the per-scenario optimization solves.
    """
    transport_module.POLICY_FILE = POLICY_FILE
    years = params["years"]
    methods = params["methods"]
    _, capacity, producer_iso, country_meta = load_inputs(params["collection_scenario"])
    _WORKER_CTX.clear()
    _WORKER_CTX.update(
        {
            "capacity": capacity,
            "producer_iso": producer_iso,
            "country_meta": country_meta,
            "scrap_by_type": load_scrap_by_type(params["collection_scenario"]),
            "countries": pd.read_csv(ROOT / "all_countries.csv"),
            "distance": load_distance_matrix(),
            "li_content": load_li_content(),
            "base_recovery": load_recovery_efficiency(
                params["recovery_scenario"], years, methods
            ),
            "emission": load_emission_factor(),
            "lithium_price": load_lithium_price(params["price_scenario"]),
            "policies": params["policies"],
            "run_policies": list(
                dict.fromkeys(["route_access_open"] + params["policies"])
            ),
            "years": years,
            "methods": methods,
            "strategy": params["strategy"],
            "waste_class": params["waste_class"],
            "treatment_type": params["treatment_type"],
            "delay_cost": params["delay_cost"],
            "capacity_expansion_countries": params["capacity_expansion_countries"],
        }
    )


def _run_one_scenario(item):
    """Run one mitigation scenario across all years/policies using _WORKER_CTX.

    Returns (routes_df, summary_rows) for that single scenario.
    """
    scenario_name, config = item
    ctx = _WORKER_CTX
    years = ctx["years"]
    methods = ctx["methods"]
    strategy = ctx["strategy"]
    waste_class = ctx["waste_class"]
    treatment_type = ctx["treatment_type"]
    delay_cost = ctx["delay_cost"]
    run_policies = ctx["run_policies"]
    policies = ctx["policies"]
    li_content = ctx["li_content"]
    emission = ctx["emission"]

    recovery = apply_direct_recovery_floor(
        ctx["base_recovery"], years, config.get("direct_recovery_floor")
    )
    recovery = apply_recovery_floor_by_method(
        recovery, years, config.get("recovery_floor_by_method")
    )
    scenario_lithium_price = scale_lithium_price(
        ctx["lithium_price"], config.get("lithium_price_multiplier")
    )

    # #1 Black-mass factor: add "_BM" pseudo-methods when enabled.
    black_mass = bool(config.get("black_mass", False))
    solve_methods = list(methods)
    solve_emission = emission
    if black_mass:
        recovery, solve_emission = extend_recovery_emission_with_black_mass(
            recovery, emission, methods, years
        )
        solve_methods = methods + [m + BLACK_MASS_SUFFIX for m in methods]

    scenario_routes = []
    summary_rows = []
    for year in years:
        supply = make_supply_by_type(
            ctx["scrap_by_type"], ctx["countries"], year, strategy
        )
        potential_li = potential_lithium_t(supply, li_content)
        destination_capacity = make_capacity(ctx["capacity"], ctx["producer_iso"], year)
        destination_capacity = expand_key_capacity(
            destination_capacity,
            ctx["capacity_expansion_countries"],
            config.get("capacity_multiplier"),
        )
        year_routes = []
        for policy in run_policies:
            policy_for_cost = config.get("policy_override")
            if policy_for_cost is None:
                policy_for_cost = (
                    "open_policy" if policy == "route_access_open" else policy
                )
            method_costs = build_method_costs(
                ctx["distance"],
                destination_capacity,
                ctx["country_meta"],
                methods,
                policy_for_cost,
                year,
                waste_class,
                treatment_type,
                delay_cost,
            )
            method_components = build_method_cost_components(
                ctx["distance"],
                destination_capacity,
                ctx["country_meta"],
                methods,
                policy_for_cost,
                year,
                waste_class,
                treatment_type,
                delay_cost,
            )
            apply_direct_cost_multiplier(
                method_costs,
                method_components,
                config.get("direct_cost_multiplier"),
            )
            if black_mass:
                bm_costs, bm_components = build_black_mass_costs(
                    ctx["distance"],
                    destination_capacity,
                    ctx["country_meta"],
                    methods,
                    year,
                    waste_class,
                    treatment_type,
                    delay_cost,
                )
                method_costs.update(bm_costs)
                method_components.update(bm_components)
            routes = solve_joint_scenario(
                supply,
                destination_capacity,
                method_costs,
                solve_methods,
                method_components,
                year,
                li_content,
                recovery,
                scenario_lithium_price,
                lithium_aware=bool(config.get("lithium_aware", False)),
                objective=config.get("objective", "cost_min"),
            )
            routes = add_lithium_outputs(
                routes, year, recovery, li_content, solve_emission
            )
            routes["year"] = year
            routes["policy_scenario"] = policy
            routes["cost_policy_scenario"] = policy_for_cost
            routes["strategy"] = strategy
            routes["mitigation_scenario"] = scenario_name
            routes["scenario_description"] = config["description"]
            year_routes.append(routes)
        year_routes = pd.concat(year_routes, ignore_index=True)
        scenario_routes.append(year_routes)

        for policy in policies:
            policy_routes = year_routes[year_routes["policy_scenario"] == policy]
            summary = summarize_routes(policy_routes, potential_li)
            summary.update(
                {
                    "mitigation_scenario": scenario_name,
                    "scenario_description": config["description"],
                    "year": year,
                    "policy_scenario": policy,
                    "strategy": strategy,
                    "route_access_loss_t": route_access_loss(year_routes, policy),
                    "route_access_displaced_lithium_t": route_access_displaced_lithium(
                        year_routes, policy
                    ),
                }
            )
            summary_rows.append(summary)

    return pd.concat(scenario_routes, ignore_index=True), summary_rows


def run_scenarios(
    collection_scenario,
    recovery_scenario,
    policies,
    years,
    methods,
    strategy,
    waste_class,
    treatment_type,
    delay_cost,
    price_scenario,
    capacity_expansion_countries,
    include_max_li,
    run_pareto,
    jobs,
):
    transport_module.POLICY_FILE = POLICY_FILE
    SCENARIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    active_scenarios = {
        name: config
        for name, config in SCENARIOS.items()
        if include_max_li or name != "max_lithium"
    }
    scenario_items = list(active_scenarios.items())

    worker_params = {
        "collection_scenario": collection_scenario,
        "recovery_scenario": recovery_scenario,
        "price_scenario": price_scenario,
        "policies": policies,
        "years": years,
        "methods": methods,
        "strategy": strategy,
        "waste_class": waste_class,
        "treatment_type": treatment_type,
        "delay_cost": delay_cost,
        "capacity_expansion_countries": capacity_expansion_countries,
    }

    n_jobs = max(1, min(int(jobs), len(scenario_items)))
    if n_jobs == 1:
        _init_worker(worker_params)
        results = [_run_one_scenario(item) for item in scenario_items]
    else:
        mp_context = multiprocessing.get_context("spawn")
        with mp_context.Pool(
            processes=n_jobs,
            initializer=_init_worker,
            initargs=(worker_params,),
        ) as pool:
            results = pool.map(_run_one_scenario, scenario_items)
    print(f"Ran {len(scenario_items)} scenarios across {n_jobs} worker(s).")

    all_routes = [routes_df for routes_df, _ in results]
    summary_rows = [row for _, rows in results for row in rows]

    routes = pd.concat(all_routes, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    baseline = summary[summary["mitigation_scenario"] == "baseline"][
        ["year", "policy_scenario", "total_lithium_loss_t"]
    ].rename(columns={"total_lithium_loss_t": "baseline_total_lithium_loss_t"})
    summary = summary.merge(baseline, on=["year", "policy_scenario"], how="left")
    summary["loss_reduction_vs_baseline_t"] = (
        summary["baseline_total_lithium_loss_t"] - summary["total_lithium_loss_t"]
    )
    summary["loss_reduction_vs_baseline_pct"] = np.where(
        summary["baseline_total_lithium_loss_t"] > 0,
        summary["loss_reduction_vs_baseline_t"]
        / summary["baseline_total_lithium_loss_t"]
        * 100.0,
        0.0,
    )

    routes_output = SCENARIO_OUTPUT_DIR / "lithium_loss_scenarios_routes.csv"
    summary_output = SCENARIO_OUTPUT_DIR / "lithium_loss_scenarios_summary.csv"
    pareto_output = SCENARIO_OUTPUT_DIR / "lithium_pareto_frontier.csv"
    routes.to_csv(routes_output, index=False)
    summary.to_csv(summary_output, index=False)
    if run_pareto:
        _, capacity, producer_iso, country_meta = load_inputs(collection_scenario)
        scrap_by_type = load_scrap_by_type(collection_scenario)
        countries = pd.read_csv(ROOT / "all_countries.csv")
        distance = load_distance_matrix()
        li_content = load_li_content()
        base_recovery = load_recovery_efficiency(recovery_scenario, years, methods)
        emission = load_emission_factor()
        lithium_price = load_lithium_price(price_scenario)
        pareto = run_pareto_frontier(
            capacity,
            producer_iso,
            country_meta,
            scrap_by_type,
            countries,
            distance,
            li_content,
            base_recovery,
            emission,
            lithium_price,
            policies,
            years,
            methods,
            strategy,
            waste_class,
            treatment_type,
            delay_cost,
            capacity_expansion_countries,
        )
        pareto.to_csv(pareto_output, index=False)
        return routes_output, summary_output, pareto_output
    return routes_output, summary_output, None


def main():
    parser = argparse.ArgumentParser(
        description="Run lithium-loss mitigation scenarios for joint policy-route-technology optimization."
    )
    parser.add_argument("--collection-scenario", default="high_collection")
    parser.add_argument("--recovery-scenario", default="baseline")
    parser.add_argument(
        "--policies",
        default="current_policy,reference_policy,strict_policy,critical_route_policy",
    )
    parser.add_argument("--years", default="2030,2040,2050")
    parser.add_argument("--methods", default="Direct,Hydro,Pyro")
    parser.add_argument("--strategy", default="Strategy 3")
    parser.add_argument("--waste-class", default="hazardous")
    parser.add_argument("--treatment-type", default="recovery")
    parser.add_argument("--delay-cost", type=float, default=DEFAULT_DELAY_COST_USD_PER_T_DAY)
    parser.add_argument("--price-scenario", default="baseline")
    parser.add_argument(
        "--capacity-expansion-countries",
        default="CHN,KOR,JPN,USA,IND",
        help="Comma-separated ISO3 destination countries expanded in the capacity scenario.",
    )
    parser.add_argument(
        "--include-max-li",
        action="store_true",
        help="Include the heavier max_lithium upper-bound scenario in the main scenario CSVs.",
    )
    parser.add_argument(
        "--run-pareto",
        action="store_true",
        help="Also write the lightweight 2050 cost-recovery Pareto frontier CSV.",
    )
    parser.add_argument(
        "--pareto-only",
        action="store_true",
        help="Only write the 2050 cost-recovery Pareto frontier CSV.",
    )
    parser.add_argument(
        "--factorial",
        action="store_true",
        help=(
            "Run the full-factorial mitigation set (baseline + all 63 factor "
            "combinations) and write to a separate lithium_loss_factorial/ folder."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Worker processes for the scenario sweep. 1 = sequential (debug).",
    )
    args = parser.parse_args()

    if args.factorial:
        global SCENARIOS, SCENARIO_OUTPUT_DIR
        SCENARIOS = build_factorial_scenarios()
        SCENARIO_OUTPUT_DIR = OUTPUT_DIR / "lithium_loss_factorial"
        print(f"Factorial mode: {len(SCENARIOS)} scenarios -> {SCENARIO_OUTPUT_DIR}")

    if args.pareto_only:
        transport_module.POLICY_FILE = POLICY_FILE
        _, capacity, producer_iso, country_meta = load_inputs(args.collection_scenario)
        scrap_by_type = load_scrap_by_type(args.collection_scenario)
        countries = pd.read_csv(ROOT / "all_countries.csv")
        years = parse_years(args.years)
        methods = parse_csv(args.methods)
        distance = load_distance_matrix()
        li_content = load_li_content()
        base_recovery = load_recovery_efficiency(args.recovery_scenario, years, methods)
        emission = load_emission_factor()
        lithium_price = load_lithium_price(args.price_scenario)
        SCENARIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        pareto = run_pareto_frontier(
            capacity,
            producer_iso,
            country_meta,
            scrap_by_type,
            countries,
            distance,
            li_content,
            base_recovery,
            emission,
            lithium_price,
            parse_csv(args.policies),
            years,
            methods,
            args.strategy,
            args.waste_class,
            args.treatment_type,
            args.delay_cost,
            parse_csv(args.capacity_expansion_countries),
        )
        pareto_output = SCENARIO_OUTPUT_DIR / "lithium_pareto_frontier.csv"
        pareto.to_csv(pareto_output, index=False)
        print(f"Wrote {pareto_output}")
        return

    routes_output, summary_output, pareto_output = run_scenarios(
        args.collection_scenario,
        args.recovery_scenario,
        parse_csv(args.policies),
        parse_years(args.years),
        parse_csv(args.methods),
        args.strategy,
        args.waste_class,
        args.treatment_type,
        args.delay_cost,
        args.price_scenario,
        parse_csv(args.capacity_expansion_countries),
        args.include_max_li,
        args.run_pareto,
        args.jobs,
    )
    print(f"Wrote {routes_output}")
    print(f"Wrote {summary_output}")
    if pareto_output is not None:
        print(f"Wrote {pareto_output}")


if __name__ == "__main__":
    main()
