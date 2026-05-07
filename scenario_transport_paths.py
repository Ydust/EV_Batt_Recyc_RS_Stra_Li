import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parent
SCENARIO_SCRAP_FILE = (
    ROOT / "Scenario result" / "recycling_rate" / "EV_battery_inuse_scrap_collected_by_scenario.csv"
)
CAPACITY_FILE = ROOT / "recycling_cap_2050.csv"
DIST_FILE = ROOT / "dist_nation.csv"
COUNTRY_FILE = ROOT / "nation_list_new.csv"
ALL_COUNTRIES_FILE = ROOT / "all_countries.csv"
COST_FILE = ROOT / "cost" / "cost_coun_df.csv"
CARBON_TAX_FILE = ROOT / "cost" / "carbon_tax_nation.csv"
EMISSION_FILE = ROOT / "cost" / "recycling_emission_count.csv"
POLICY_FILE = ROOT / "waste_trade_policy_constraints.csv"

BIG_M = 1e6
FORBIDDEN_COST = 1e12
DEFAULT_DELAY_COST_USD_PER_T_DAY = 1.0

ANNEX_VII_COUNTRIES = {
    "Australia",
    "Austria",
    "Belgium",
    "Canada",
    "Czechia",
    "Denmark",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Iceland",
    "Ireland",
    "Italy",
    "Japan",
    "Korea",
    "Luxembourg",
    "Netherlands",
    "New Zealand",
    "Norway",
    "Poland",
    "Portugal",
    "Slovakia",
    "Spain",
    "Sweden",
    "Switzerland",
    "Turkey",
    "United Kingdom",
    "USA",
}

BASEL_NON_PARTIES = {
    "Haiti",
    "San Marino",
    "South Sudan",
    "Timor-Leste",
    "USA",
}


def solve_transport(cost_matrix, supply, demand):
    supply = supply[supply["scrap"] > 0].copy()
    demand = demand[demand["Mass_v"] > 0].copy()
    supply = supply[supply.index.isin(cost_matrix.index)].copy()
    demand = demand[demand.index.isin(cost_matrix.columns)].copy()
    if supply.empty or demand.empty:
        return pd.DataFrame()

    supply_nodes = supply.index.tolist()
    demand_nodes = demand.index.tolist()
    cost_matrix = cost_matrix.reindex(index=supply_nodes, columns=demand_nodes, fill_value=BIG_M)
    cost = cost_matrix.loc[supply_nodes, demand_nodes].fillna(BIG_M).values
    supply_values = supply["scrap"].values
    demand_values = demand["Mass_v"].values

    difference = demand_values.sum() - supply_values.sum()
    if difference > 0:
        supply_values = np.append(supply_values, difference)
        cost = np.vstack([cost, np.full((1, len(demand_nodes)), BIG_M)])
        supply_nodes.append("Virtual_Supply")
    elif difference < 0:
        demand_values = np.append(demand_values, -difference)
        cost = np.hstack([cost, np.full((len(supply_nodes), 1), BIG_M)])
        demand_nodes.append("Virtual_Demand")

    c = cost.flatten()
    bounds = [(0, 0) if value >= FORBIDDEN_COST else (0, None) for value in c]
    n_supply = len(supply_values)
    n_demand = len(demand_values)
    a_eq = []
    b_eq = []
    for i in range(n_supply):
        a_eq.append([1 if j // n_demand == i else 0 for j in range(n_supply * n_demand)])
        b_eq.append(supply_values[i])
    for j in range(n_demand):
        a_eq.append([1 if j == k % n_demand else 0 for k in range(n_supply * n_demand)])
        b_eq.append(demand_values[j])

    result = linprog(c, A_eq=np.array(a_eq), b_eq=np.array(b_eq), bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(result.message)
    return pd.DataFrame(result.x.reshape(n_supply, n_demand), index=supply_nodes, columns=demand_nodes)


def solve_transport_with_policy_fallback(cost_matrix, supply, demand):
    try:
        return solve_transport(cost_matrix, supply, demand)
    except RuntimeError as exc:
        if "infeasible" not in str(exc).lower():
            raise
        relaxed = cost_matrix.copy()
        supply_aug = supply.copy()
        demand_aug = demand.copy()
        real_supply_total = float(supply_aug["scrap"].sum())
        real_demand_total = float(demand_aug["Mass_v"].sum())

        if "Virtual_Supply" not in relaxed.index:
            relaxed.loc["Virtual_Supply"] = BIG_M
        if "Virtual_Demand" not in relaxed.columns:
            relaxed["Virtual_Demand"] = BIG_M
        relaxed.loc["Virtual_Supply", "Virtual_Demand"] = 0.0

        supply_aug.loc["Virtual_Supply", "scrap"] = real_demand_total
        demand_aug.loc["Virtual_Demand", "Mass_v"] = real_supply_total
        return solve_transport(relaxed, supply_aug, demand_aug)


def load_distance_matrix():
    dist = pd.read_csv(DIST_FILE)
    dist = dist[["iso_o", "iso_d", "distcap"]].copy()
    dist["distcap"] = dist["distcap"] * 0.03 / 1e3
    return dist.pivot_table(index="iso_o", columns="iso_d", values="distcap", aggfunc="min")


def load_inputs(collection_scenario):
    scrap = pd.read_csv(SCENARIO_SCRAP_FILE)
    scrap = scrap[scrap["scenario"] == collection_scenario].copy()
    if scrap.empty:
        raise ValueError(f"No collection scenario found: {collection_scenario}")
    scrap = scrap.groupby(["Year", "region"], as_index=False)["scrap"].sum()

    countries = pd.read_csv(COUNTRY_FILE)[["region", "iso3"]].rename(columns={"region": "country"})
    scrap = scrap.rename(columns={"region": "country"}).merge(countries, on="country", how="left")
    scrap = scrap.dropna(subset=["iso3"])

    capacity = pd.read_csv(CAPACITY_FILE)
    capacity = capacity.melt(id_vars="Year", var_name="country", value_name="Mass_v")
    capacity["Mass_v"] = capacity["Mass_v"] * 10000
    capacity = capacity.merge(countries, on="country", how="left").dropna(subset=["iso3"])

    all_countries = pd.read_csv(ALL_COUNTRIES_FILE)
    producers = set(all_countries.loc[all_countries["producer"] == True, "country"])
    producer_iso = set(countries.loc[countries["country"].isin(producers), "iso3"])
    country_meta = all_countries[["country", "iso3", "continent"]].dropna(subset=["iso3"]).copy()
    return scrap, capacity, producer_iso, country_meta


def load_recycling_unit_cost(method, countries):
    cost = pd.read_csv(COST_FILE)
    if method not in cost.columns:
        raise ValueError(f"No recycling cost column found for method: {method}")
    cost = cost[["country", "Recycling_capacity", method]].dropna()
    unit_cost = {}
    for country in countries["country"].dropna().unique():
        country_cost = cost[cost["country"] == country].sort_values("Recycling_capacity")
        if country_cost.empty:
            continue
        # Use the median point as a stable linearized unit-cost proxy for the LP.
        unit_cost[country] = float(country_cost[method].median())
    if not unit_cost:
        return pd.Series(0.0, index=countries["iso3"].unique())
    global_default = float(pd.Series(unit_cost).median())
    countries = countries.copy()
    countries["unit_recycling_cost"] = countries["country"].map(unit_cost).fillna(global_default)
    return countries.set_index("iso3")["unit_recycling_cost"]


def load_carbon_unit_cost(method, countries):
    carbon = pd.read_csv(CARBON_TAX_FILE)
    carbon = carbon[carbon["Metric"] == "US$/tCO2e"].copy()
    carbon_price = carbon.groupby("Jurisdiction Covered")["Value"].mean()

    emission = pd.read_csv(EMISSION_FILE)
    method_emission = emission[emission["recycling_m"] == method]["CO2_new"].mean()
    if pd.isna(method_emission):
        return pd.Series(0.0, index=countries["iso3"].unique())

    # CO2_new is g/kg battery, numerically equivalent to kg/t battery.
    # Convert carbon $/tCO2 to modeled $/kg battery to match existing transport/cost units.
    countries = countries.copy()
    countries["carbon_price"] = countries["country"].map(carbon_price).fillna(0.0)
    countries["unit_carbon_cost"] = countries["carbon_price"] * (float(method_emission) / 1000.0) / 1000.0
    return countries.set_index("iso3")["unit_carbon_cost"]


def build_country_groups(country_meta):
    countries = set(country_meta["country"].dropna())
    annex_vii = countries & ANNEX_VII_COUNTRIES
    basel_non_parties = countries & BASEL_NON_PARTIES
    basel_parties = countries - basel_non_parties
    return {
        "ALL": set(country_meta["iso3"]),
        "AnnexVII": set(country_meta.loc[country_meta["country"].isin(annex_vii), "iso3"]),
        "NonAnnexVII": set(country_meta.loc[~country_meta["country"].isin(annex_vii), "iso3"]),
        "BaselParty": set(country_meta.loc[country_meta["country"].isin(basel_parties), "iso3"]),
        "BaselNonParty": set(country_meta.loc[country_meta["country"].isin(basel_non_parties), "iso3"]),
        "EU": set(country_meta.loc[country_meta["continent"] == "Europe", "iso3"]),
        "NonOECD": set(country_meta.loc[~country_meta["country"].isin(ANNEX_VII_COUNTRIES), "iso3"]),
    }


def select_policy_nodes(rule, side, country_meta, groups):
    country_col = f"{side}_country"
    group_col = f"{side}_group"
    country_value = rule.get(country_col)
    group_value = rule.get(group_col)

    if pd.notna(country_value) and str(country_value).strip():
        match = country_meta[country_meta["country"] == str(country_value).strip()]
        return set(match["iso3"])
    if pd.notna(group_value) and str(group_value).strip():
        return set(groups.get(str(group_value).strip(), set()))
    return set(groups["ALL"])


def apply_policy_constraints(cost_matrix, country_meta, policy_scenario, year, waste_class, treatment_type, delay_cost):
    if not POLICY_FILE.exists():
        return cost_matrix

    policy = pd.read_csv(POLICY_FILE)
    policy = policy[
        (policy["scenario"] == policy_scenario)
        & (policy["start_year"] <= year)
        & (policy["end_year"] >= year)
        & (policy["waste_class"].isin([waste_class, "ALL"]))
        & (policy["treatment_type"].isin([treatment_type, "ALL"]))
    ].copy()
    if policy.empty:
        return cost_matrix

    groups = build_country_groups(country_meta)
    adjusted = cost_matrix.copy()
    for _, rule in policy.iterrows():
        source_nodes = select_policy_nodes(rule, "source", country_meta, groups) & set(adjusted.index)
        destination_nodes = select_policy_nodes(rule, "destination", country_meta, groups) & set(adjusted.columns)
        if not source_nodes or not destination_nodes:
            continue
        route_pairs = [(src, dst) for src in source_nodes for dst in destination_nodes if src != dst]
        if not route_pairs:
            continue
        if int(rule["forbidden"]) == 1:
            for src, dst in route_pairs:
                adjusted.loc[src, dst] = FORBIDDEN_COST
        else:
            penalty = float(rule.get("policy_penalty_usd_per_t", 0) or 0) / 1000.0
            delay = float(rule.get("approval_delay_days", 0) or 0) * delay_cost / 1000.0
            for src, dst in route_pairs:
                adjusted.loc[src, dst] += penalty + delay
    return adjusted


def policy_route_diagnostics(cost_matrix, country_meta, policy_scenario, year, waste_class, treatment_type, delay_cost):
    columns = [
        "source_iso3",
        "destination_iso3",
        "scenario",
        "forbidden",
        "policy_penalty_usd_per_t",
        "approval_delay_days",
        "modeled_policy_cost_per_kg",
        "source_basis",
        "notes",
    ]
    if not POLICY_FILE.exists():
        return pd.DataFrame(columns=columns)

    policy = pd.read_csv(POLICY_FILE)
    policy = policy[
        (policy["scenario"] == policy_scenario)
        & (policy["start_year"] <= year)
        & (policy["end_year"] >= year)
        & (policy["waste_class"].isin([waste_class, "ALL"]))
        & (policy["treatment_type"].isin([treatment_type, "ALL"]))
    ].copy()
    if policy.empty:
        return pd.DataFrame(columns=columns)

    groups = build_country_groups(country_meta)
    rows = []
    for _, rule in policy.iterrows():
        source_nodes = select_policy_nodes(rule, "source", country_meta, groups) & set(cost_matrix.index)
        destination_nodes = select_policy_nodes(rule, "destination", country_meta, groups) & set(cost_matrix.columns)
        for src in source_nodes:
            for dst in destination_nodes:
                if src == dst:
                    continue
                penalty = float(rule.get("policy_penalty_usd_per_t", 0) or 0)
                delay_days = float(rule.get("approval_delay_days", 0) or 0)
                rows.append(
                    {
                        "source_iso3": src,
                        "destination_iso3": dst,
                        "scenario": policy_scenario,
                        "forbidden": int(rule["forbidden"]),
                        "policy_penalty_usd_per_t": penalty,
                        "approval_delay_days": delay_days,
                        "modeled_policy_cost_per_kg": (
                            penalty + delay_days * delay_cost
                        )
                        / 1000.0,
                        "source_basis": rule.get("source_basis", ""),
                        "notes": rule.get("notes", ""),
                    }
                )
    return pd.DataFrame(rows, columns=columns)


def write_cost_diagnostics(
    output_dir,
    method,
    year,
    distance,
    demand,
    country_meta,
    supply_strategy2,
    supply_strategy3,
    policy_scenario,
    waste_class,
    treatment_type,
    delay_cost,
):
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    destination_countries = country_meta[country_meta["iso3"].isin(demand.index)].copy()
    recycling_cost = load_recycling_unit_cost(method, destination_countries)
    carbon_cost = load_carbon_unit_cost(method, destination_countries)

    recycling_cost.rename("recycling_unit_cost").reset_index().rename(
        columns={"index": "destination_iso3"}
    ).to_csv(diagnostics_dir / f"recycling_cost_by_destination_{method}.csv", index=False)
    carbon_cost.rename("carbon_unit_cost").reset_index().rename(
        columns={"index": "destination_iso3"}
    ).to_csv(diagnostics_dir / f"carbon_cost_by_destination_{method}.csv", index=False)

    route_cost = distance.reindex(columns=demand.index).stack().reset_index()
    route_cost.columns = ["source_iso3", "destination_iso3", "transport_cost_per_kg"]
    route_cost.to_csv(diagnostics_dir / f"transport_cost_by_route_{method}.csv", index=False)

    policy_diag = policy_route_diagnostics(
        distance.reindex(columns=demand.index, fill_value=BIG_M),
        country_meta,
        policy_scenario,
        year,
        waste_class,
        treatment_type,
        delay_cost,
    )
    policy_diag.to_csv(diagnostics_dir / f"policy_penalty_by_route_{method}.csv", index=False)
    policy_diag[policy_diag["forbidden"] == 1].to_csv(
        diagnostics_dir / f"forbidden_routes_{method}.csv", index=False
    )

    supply = (
        supply_strategy3.rename(columns={"scrap": "strategy3_supply"})
        .join(supply_strategy2.rename(columns={"scrap": "strategy2_supply"}), how="left")
        .fillna(0)
        .reset_index()
        .rename(columns={"iso3": "source_iso3"})
    )
    blocked = policy_diag[policy_diag["forbidden"] == 1].merge(
        supply, on="source_iso3", how="left"
    )
    blocked.to_csv(
        diagnostics_dir / f"blocked_or_rerouted_scrap_{method}.csv", index=False
    )


def build_total_cost_matrix(distance, demand, country_meta, method, policy_scenario, year, waste_class, treatment_type, delay_cost):
    destination_countries = country_meta[country_meta["iso3"].isin(demand.index)].copy()
    recycling_cost = load_recycling_unit_cost(method, destination_countries)
    carbon_cost = load_carbon_unit_cost(method, destination_countries)
    destination_cost = (recycling_cost.add(carbon_cost, fill_value=0.0)).reindex(demand.index).fillna(0.0)

    total_cost = distance.reindex(columns=demand.index, fill_value=BIG_M).copy()
    total_cost = total_cost.add(destination_cost, axis="columns")
    total_cost = apply_policy_constraints(
        total_cost,
        country_meta,
        policy_scenario,
        year,
        waste_class,
        treatment_type,
        delay_cost,
    )
    return total_cost


def write_empty_transport_log(output_dir):
    pd.DataFrame(
        columns=["From Region", "To Region", "Amount Transported", "Remaining Storage"]
    ).to_csv(output_dir / "transport_log.csv", index=False)


def generate_paths(
    collection_scenario,
    recovery_scenario,
    years,
    methods,
    policy_scenario="baseline",
    waste_class="hazardous",
    treatment_type="recovery",
    delay_cost=DEFAULT_DELAY_COST_USD_PER_T_DAY,
):
    scrap, capacity, producer_iso, country_meta = load_inputs(collection_scenario)
    distance = load_distance_matrix()
    output_root = ROOT / "trans" / "scenario_paths" / collection_scenario / recovery_scenario

    for year in years:
        output_dir = output_root / str(year)
        output_dir.mkdir(parents=True, exist_ok=True)

        year_scrap = scrap[scrap["Year"] == year].copy()
        year_capacity = capacity[capacity["Year"] == year].copy()
        demand = year_capacity[year_capacity["iso3"].isin(producer_iso)][["iso3", "Mass_v"]].copy()
        demand = demand.groupby("iso3", as_index=True)["Mass_v"].sum().to_frame()

        supply_strategy2 = year_scrap[year_scrap["iso3"].isin(producer_iso)][["iso3", "scrap"]].copy()
        supply_strategy2 = supply_strategy2.groupby("iso3", as_index=True)["scrap"].sum().to_frame()

        supply_strategy3 = year_scrap[["iso3", "scrap"]].copy()
        supply_strategy3 = supply_strategy3.groupby("iso3", as_index=True)["scrap"].sum().to_frame()

        for method in methods:
            write_cost_diagnostics(
                output_dir,
                method,
                year,
                distance,
                demand,
                country_meta,
                supply_strategy2,
                supply_strategy3,
                policy_scenario,
                waste_class,
                treatment_type,
                delay_cost,
            )
            total_cost = build_total_cost_matrix(
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
            plan2 = solve_transport_with_policy_fallback(total_cost, supply_strategy2, demand)
            plan3 = solve_transport_with_policy_fallback(total_cost, supply_strategy3, demand)
            plan2.to_csv(output_dir / f"Strategy 2_{method}_optimal_transportation_plan.csv")
            plan3.to_csv(output_dir / f"Strategy 3_{method}_optimal_transportation_plan.csv")
        write_empty_transport_log(output_dir)
    return output_root


def main():
    parser = argparse.ArgumentParser(description="Generate scenario-specific Strategy 2/3 transport paths.")
    parser.add_argument("--collection-scenario", default="high_collection")
    parser.add_argument("--recovery-scenario", default="baseline")
    parser.add_argument("--policy-scenario", default="baseline")
    parser.add_argument("--waste-class", default="hazardous")
    parser.add_argument("--treatment-type", default="recovery")
    parser.add_argument("--delay-cost", type=float, default=DEFAULT_DELAY_COST_USD_PER_T_DAY)
    parser.add_argument("--years", default="2025,2030,2035,2040,2045,2050")
    parser.add_argument("--methods", default="Pyro,Hydro,Direct")
    args = parser.parse_args()

    years = [int(y.strip()) for y in args.years.split(",") if y.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    output = generate_paths(
        args.collection_scenario,
        args.recovery_scenario,
        years,
        methods,
        policy_scenario=args.policy_scenario,
        waste_class=args.waste_class,
        treatment_type=args.treatment_type,
        delay_cost=args.delay_cost,
    )
    print(output)


if __name__ == "__main__":
    main()
