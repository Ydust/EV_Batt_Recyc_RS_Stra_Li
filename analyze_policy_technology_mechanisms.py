import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import run_lithium_loss_scenarios as scenarios
import scenario_transport_paths as transport_module
from joint_policy_transport_technology_optimization import (
    build_method_cost_components,
    build_method_costs,
)
from scenario_transport_paths import DEFAULT_DELAY_COST_USD_PER_T_DAY, load_distance_matrix, load_inputs


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "Figure_data" / "joint_policy_technology" / "policy_technology_mechanisms"
ROUTE_FILE = ROOT / "Figure_data" / "joint_policy_technology" / "joint_policy_transport_technology_routes.csv"
ELASTICITY_DIR = (
    ROOT
    / "Figure_data"
    / "joint_policy_technology"
    / "policy_objective_technology_elasticity"
)
METHODS = ["Direct", "Hydro", "Pyro"]
POLICIES = ["reference_policy", "current_policy", "strict_policy", "critical_route_policy"]
YEARS = [2030, 2040, 2050]
STRATEGY = "Strategy 3"
WASTE_CLASS = "hazardous"
TREATMENT_TYPE = "recovery"
CAPACITY_EXPANSION_COUNTRIES = ["CHN", "KOR", "JPN", "USA", "IND"]
RECOVERY_FLOOR_BY_METHOD = {"Direct": 0.97, "Hydro": 0.95, "Pyro": 0.35}


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_years(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def real_routes(routes):
    return routes[~routes["is_unprocessed"].astype(str).str.lower().eq("true")].copy()


def route_modeled_cost(routes):
    real = real_routes(routes)
    return float(real[["transport_cost", "recycling_cost", "carbon_cost", "policy_cost"]].sum(axis=1).sum())


def add_numeric_routes(routes):
    numeric = [
        "scrap_t",
        "contained_lithium_t",
        "recovered_lithium_t",
        "transport_cost",
        "recycling_cost",
        "carbon_cost",
        "policy_cost",
    ]
    for column in numeric:
        if column in routes.columns:
            routes[column] = pd.to_numeric(routes[column], errors="coerce").fillna(0.0)
    routes["year"] = pd.to_numeric(routes["year"], errors="coerce").astype(int)
    return routes


def technology_accessible_li(years, policies, solver_methods):
    transport_module.POLICY_FILE = scenarios.POLICY_FILE
    _, capacity, producer_iso, country_meta = load_inputs("high_collection")
    scrap_by_type = scenarios.load_scrap_by_type("high_collection")
    countries = pd.read_csv(ROOT / "all_countries.csv")
    distance = load_distance_matrix()
    li_content = scenarios.load_li_content()
    recovery = scenarios.load_recovery_efficiency("baseline", years, METHODS)
    recovery = scenarios.apply_recovery_floor_by_method(recovery, years, RECOVERY_FLOOR_BY_METHOD)
    emission = scenarios.load_emission_factor()
    lithium_price = scenarios.load_lithium_price("baseline")
    rows = []
    route_rows = []
    for year in years:
        supply = scenarios.make_supply_by_type(scrap_by_type, countries, year, STRATEGY)
        destination_capacity = scenarios.make_capacity(capacity, producer_iso, year)
        destination_capacity = scenarios.expand_key_capacity(
            destination_capacity, CAPACITY_EXPANSION_COUNTRIES, 1.25
        )
        potential_li = scenarios.potential_lithium_t(supply, li_content)
        for policy in policies:
            method_costs_all = build_method_costs(
                distance,
                destination_capacity,
                country_meta,
                METHODS,
                policy,
                year,
                WASTE_CLASS,
                TREATMENT_TYPE,
                DEFAULT_DELAY_COST_USD_PER_T_DAY,
            )
            method_components_all = build_method_cost_components(
                distance,
                destination_capacity,
                country_meta,
                METHODS,
                policy,
                year,
                WASTE_CLASS,
                TREATMENT_TYPE,
                DEFAULT_DELAY_COST_USD_PER_T_DAY,
            )
            for method in METHODS:
                try:
                    routes = scenarios.solve_joint_scenario(
                        supply,
                        destination_capacity,
                        {method: method_costs_all[method]},
                        [method],
                        {method: method_components_all[method]},
                        year,
                        li_content,
                        recovery,
                        lithium_price,
                        objective="max_lithium",
                        solver_methods=solver_methods,
                    )
                    routes = scenarios.add_lithium_outputs(routes, year, recovery, li_content, emission)
                    recovered_li = float(real_routes(routes)["recovered_lithium_t"].sum())
                    processed_scrap = float(real_routes(routes)["scrap_t"].sum())
                    unprocessed_scrap = float(
                        routes[routes["is_unprocessed"].astype(bool)]["scrap_t"].sum()
                    )
                    modeled_cost = route_modeled_cost(routes)
                    status = "success"
                    message = ""
                    routes["year"] = year
                    routes["policy_scenario"] = policy
                    routes["technology_allowed"] = method
                    route_rows.append(routes)
                except RuntimeError as exc:
                    recovered_li = np.nan
                    processed_scrap = np.nan
                    unprocessed_scrap = np.nan
                    modeled_cost = np.nan
                    status = "failed"
                    message = str(exc)
                rows.append(
                    {
                        "year": year,
                        "policy_scenario": policy,
                        "technology_allowed": method,
                        "max_accessible_recovered_lithium_t": recovered_li,
                        "potential_lithium_t": potential_li,
                        "potential_recovery_pct": recovered_li / potential_li * 100.0
                        if potential_li > 0 and pd.notna(recovered_li)
                        else np.nan,
                        "processed_scrap_t": processed_scrap,
                        "unprocessed_scrap_t": unprocessed_scrap,
                        "route_modeled_cost": modeled_cost,
                        "solve_status": status,
                        "solve_message": message,
                    }
                )
    summary = pd.DataFrame(rows)
    reference = summary[summary["policy_scenario"] == "reference_policy"][
        ["year", "technology_allowed", "max_accessible_recovered_lithium_t"]
    ].rename(columns={"max_accessible_recovered_lithium_t": "reference_max_accessible_li_t"})
    summary = summary.merge(reference, on=["year", "technology_allowed"], how="left")
    summary["max_accessible_li_delta_vs_reference_t"] = (
        summary["max_accessible_recovered_lithium_t"] - summary["reference_max_accessible_li_t"]
    )
    summary["max_accessible_li_delta_vs_reference_pct"] = (
        summary["max_accessible_li_delta_vs_reference_t"]
        / summary["reference_max_accessible_li_t"]
        * 100.0
    )
    route_table = pd.concat(route_rows, ignore_index=True) if route_rows else pd.DataFrame()
    return summary, route_table


def destination_technology_portfolio(routes):
    real = real_routes(routes)
    group_cols = ["year", "policy_scenario", "destination_iso3", "technology"]
    grouped = real.groupby(group_cols, as_index=False).agg(
        scrap_t=("scrap_t", "sum"),
        recovered_lithium_t=("recovered_lithium_t", "sum"),
        route_count=("source_iso3", "nunique"),
    )
    totals = grouped.groupby(["year", "policy_scenario", "destination_iso3"], as_index=False).agg(
        destination_recovered_lithium_t=("recovered_lithium_t", "sum"),
        destination_scrap_t=("scrap_t", "sum"),
    )
    grouped = grouped.merge(totals, on=["year", "policy_scenario", "destination_iso3"], how="left")
    grouped["technology_share_pct"] = (
        grouped["recovered_lithium_t"] / grouped["destination_recovered_lithium_t"] * 100.0
    )
    wide = grouped.pivot_table(
        index=["year", "policy_scenario", "destination_iso3"],
        columns="technology",
        values="technology_share_pct",
        aggfunc="sum",
        fill_value=0.0,
    ).reset_index()
    for method in METHODS:
        if method not in wide.columns:
            wide[method] = 0.0
    wide = wide.merge(totals, on=["year", "policy_scenario", "destination_iso3"], how="left")
    wide["dominant_technology"] = wide[METHODS].idxmax(axis=1)
    wide["direct_capable_observed"] = wide["Direct"] > 0.0
    wide["hydro_dominated_observed"] = wide["Hydro"] >= 50.0
    return grouped, wide


def destination_shift_vs_reference(destination_wide):
    ref = destination_wide[destination_wide["policy_scenario"] == "reference_policy"][
        [
            "year",
            "destination_iso3",
            "destination_recovered_lithium_t",
            "destination_scrap_t",
            "dominant_technology",
            "Direct",
            "Hydro",
            "Pyro",
        ]
    ].rename(
        columns={
            "destination_recovered_lithium_t": "reference_destination_recovered_lithium_t",
            "destination_scrap_t": "reference_destination_scrap_t",
            "dominant_technology": "reference_dominant_technology",
            "Direct": "reference_direct_share_pct",
            "Hydro": "reference_hydro_share_pct",
            "Pyro": "reference_pyro_share_pct",
        }
    )
    out = destination_wide.merge(ref, on=["year", "destination_iso3"], how="left")
    out["destination_recovered_li_delta_vs_reference_t"] = (
        out["destination_recovered_lithium_t"] - out["reference_destination_recovered_lithium_t"].fillna(0.0)
    )
    out["destination_scrap_delta_vs_reference_t"] = (
        out["destination_scrap_t"] - out["reference_destination_scrap_t"].fillna(0.0)
    )
    out["shift_from_reference_direct_capable_to_current_hydro_dominated"] = (
        (out["reference_direct_share_pct"].fillna(0.0) > 0.0)
        & (out["Hydro"].fillna(0.0) >= 50.0)
        & (out["destination_recovered_li_delta_vs_reference_t"] > 0.0)
    )
    return out


def route_technology_coupling(routes):
    real = real_routes(routes)
    route_keys = ["year", "policy_scenario", "source_iso3", "destination_iso3"]
    route_tech = real.groupby(route_keys + ["technology"], as_index=False).agg(
        recovered_lithium_t=("recovered_lithium_t", "sum"),
        scrap_t=("scrap_t", "sum"),
    )
    route_totals = route_tech.groupby(route_keys, as_index=False).agg(
        route_recovered_lithium_t=("recovered_lithium_t", "sum"),
        route_scrap_t=("scrap_t", "sum"),
    )
    route_tech = route_tech.merge(route_totals, on=route_keys, how="left")
    route_tech["technology_share_pct"] = (
        route_tech["recovered_lithium_t"] / route_tech["route_recovered_lithium_t"] * 100.0
    )
    dominant = (
        route_tech.sort_values(["year", "policy_scenario", "source_iso3", "destination_iso3", "recovered_lithium_t"])
        .groupby(route_keys, as_index=False)
        .tail(1)
        .rename(columns={"technology": "dominant_technology"})
    )
    dominant = dominant[route_keys + ["dominant_technology", "route_recovered_lithium_t", "route_scrap_t"]]
    ref = dominant[dominant["policy_scenario"] == "reference_policy"][
        ["year", "source_iso3", "destination_iso3", "dominant_technology", "route_recovered_lithium_t", "route_scrap_t"]
    ].rename(
        columns={
            "dominant_technology": "reference_dominant_technology",
            "route_recovered_lithium_t": "reference_route_recovered_lithium_t",
            "route_scrap_t": "reference_route_scrap_t",
        }
    )
    comparison = dominant.merge(ref, on=["year", "source_iso3", "destination_iso3"], how="outer")
    comparison["policy_scenario"] = comparison["policy_scenario"].fillna("missing_under_policy")
    comparison["route_recovered_lithium_t"] = comparison["route_recovered_lithium_t"].fillna(0.0)
    comparison["reference_route_recovered_lithium_t"] = comparison[
        "reference_route_recovered_lithium_t"
    ].fillna(0.0)
    comparison["route_recovered_li_delta_vs_reference_t"] = (
        comparison["route_recovered_lithium_t"] - comparison["reference_route_recovered_lithium_t"]
    )
    comparison["route_disappeared_vs_reference"] = (
        (comparison["reference_route_recovered_lithium_t"] > 1e-9)
        & (comparison["route_recovered_lithium_t"] <= 1e-9)
    )
    comparison["route_new_vs_reference"] = (
        (comparison["reference_route_recovered_lithium_t"] <= 1e-9)
        & (comparison["route_recovered_lithium_t"] > 1e-9)
    )
    comparison["technology_switched_vs_reference"] = (
        comparison["dominant_technology"].notna()
        & comparison["reference_dominant_technology"].notna()
        & (comparison["dominant_technology"] != comparison["reference_dominant_technology"])
    )
    summary = comparison.groupby(["year", "policy_scenario"], as_index=False).agg(
        switched_route_count=("technology_switched_vs_reference", "sum"),
        disappeared_route_count=("route_disappeared_vs_reference", "sum"),
        new_route_count=("route_new_vs_reference", "sum"),
        net_recovered_li_delta_t=("route_recovered_li_delta_vs_reference_t", "sum"),
    )
    return route_tech, comparison, summary


def collect_elasticity_tables(source_dirs):
    summary_frames = []
    mix_frames = []
    for source_dir in source_dirs:
        base = ELASTICITY_DIR if source_dir in {"root", ""} else ELASTICITY_DIR / source_dir
        summary_path = base / "policy_objective_technology_elasticity_summary.csv"
        mix_path = base / "policy_objective_technology_elasticity_mix.csv"
        if not summary_path.exists() or not mix_path.exists():
            continue
        summary = pd.read_csv(summary_path)
        mix = pd.read_csv(mix_path)
        summary["source_run"] = source_dir or "root"
        mix["source_run"] = source_dir or "root"
        summary_frames.append(summary)
        mix_frames.append(mix)
    if not summary_frames:
        return pd.DataFrame(), pd.DataFrame()
    summary = pd.concat(summary_frames, ignore_index=True)
    mix = pd.concat(mix_frames, ignore_index=True)
    for table in [summary, mix]:
        if "draw_id" not in table.columns:
            table["draw_id"] = 0
        table["draw_id"] = pd.to_numeric(table["draw_id"], errors="coerce").fillna(0).astype(int)
        table["year"] = pd.to_numeric(table["year"], errors="coerce").astype(int)
        table["target_share_of_max"] = pd.to_numeric(table["target_share_of_max"], errors="coerce")
    for column in [
        "route_modeled_cost",
        "target_recovered_lithium_t",
        "global_recovered_lithium_t",
        "direct_cost_multiplier",
        "hydro_cost_multiplier",
        "pyro_cost_multiplier",
        "direct_recovery_efficiency",
        "hydro_recovery_efficiency",
        "pyro_recovery_efficiency",
    ]:
        if column in summary.columns:
            summary[column] = pd.to_numeric(summary[column], errors="coerce")
    for column in ["technology_share_pct", "recovered_lithium_t"]:
        if column in mix.columns:
            mix[column] = pd.to_numeric(mix[column], errors="coerce")
    return summary, mix


def technology_risk_sensitivity(source_dirs):
    summary, mix = collect_elasticity_tables(source_dirs)
    if summary.empty or mix.empty:
        return pd.DataFrame(), pd.DataFrame()
    direct_mix = mix[(mix["technology"] == "Direct") & (mix["scope"] == "target_region")].copy()
    merged = summary.merge(
        direct_mix[
            [
                "source_run",
                "draw_id",
                "year",
                "policy_scenario",
                "target_region",
                "target_share_of_max",
                "technology_share_pct",
                "recovered_lithium_t",
            ]
        ].rename(
            columns={
                "technology_share_pct": "direct_target_region_share_pct",
                "recovered_lithium_t": "direct_target_region_recovered_li_t",
            }
        ),
        on=["source_run", "draw_id", "year", "policy_scenario", "target_region", "target_share_of_max"],
        how="left",
    )
    group_cols = ["source_run", "year", "policy_scenario", "target_region", "target_share_of_max"]
    stats = merged.groupby(group_cols, as_index=False).agg(
        draws=("draw_id", "nunique"),
        direct_share_mean_pct=("direct_target_region_share_pct", "mean"),
        direct_share_p05_pct=("direct_target_region_share_pct", lambda x: x.quantile(0.05)),
        direct_share_p50_pct=("direct_target_region_share_pct", "median"),
        direct_share_p95_pct=("direct_target_region_share_pct", lambda x: x.quantile(0.95)),
        route_cost_mean=("route_modeled_cost", "mean"),
        route_cost_p50=("route_modeled_cost", "median"),
        route_cost_p95=("route_modeled_cost", lambda x: x.quantile(0.95)),
    )
    sensitivity_rows = []
    predictors = [
        "direct_cost_multiplier",
        "hydro_cost_multiplier",
        "pyro_cost_multiplier",
        "direct_recovery_efficiency",
        "hydro_recovery_efficiency",
        "pyro_recovery_efficiency",
    ]
    for key, group in merged.groupby(group_cols):
        if group["draw_id"].nunique() < 3:
            continue
        for predictor in predictors:
            if predictor not in group.columns or group[predictor].nunique(dropna=True) < 2:
                continue
            corr = group[[predictor, "direct_target_region_share_pct"]].corr().iloc[0, 1]
            sensitivity_rows.append(
                {
                    **dict(zip(group_cols, key)),
                    "predictor": predictor,
                    "pearson_corr_with_direct_share": corr,
                }
            )
    return stats, pd.DataFrame(sensitivity_rows)


def write_outputs(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    routes = add_numeric_routes(pd.read_csv(args.route_file))
    outputs = []

    if not args.skip_accessible_li:
        accessible, accessible_routes = technology_accessible_li(
            parse_years(args.years),
            parse_csv(args.policies),
            parse_csv(args.solver_methods),
        )
        path = output_dir / "technology_accessible_li_only_method.csv"
        accessible.to_csv(path, index=False)
        outputs.append(path)
        if not accessible_routes.empty:
            route_path = output_dir / "technology_accessible_li_only_method_routes.csv"
            accessible_routes.to_csv(route_path, index=False)
            outputs.append(route_path)

    destination_long, destination_wide = destination_technology_portfolio(routes)
    destination_shift = destination_shift_vs_reference(destination_wide)
    for frame, name in [
        (destination_long, "destination_technology_portfolio_long.csv"),
        (destination_wide, "destination_technology_portfolio.csv"),
        (destination_shift, "destination_shift_vs_reference.csv"),
    ]:
        path = output_dir / name
        frame.to_csv(path, index=False)
        outputs.append(path)

    route_tech, route_comparison, route_summary = route_technology_coupling(routes)
    for frame, name in [
        (route_tech, "source_destination_technology_coupling_long.csv"),
        (route_comparison, "route_technology_switches_vs_reference.csv"),
        (route_summary, "route_technology_switch_summary.csv"),
    ]:
        path = output_dir / name
        frame.to_csv(path, index=False)
        outputs.append(path)

    risk_stats, risk_sensitivity = technology_risk_sensitivity(parse_csv(args.elasticity_source_dirs))
    if not risk_stats.empty:
        path = output_dir / "technology_risk_sensitivity_stats.csv"
        risk_stats.to_csv(path, index=False)
        outputs.append(path)
    if not risk_sensitivity.empty:
        path = output_dir / "technology_risk_sensitivity_correlations.csv"
        risk_sensitivity.to_csv(path, index=False)
        outputs.append(path)
    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Generate policy-technology mechanism indicators beyond technology shares."
    )
    parser.add_argument("--route-file", default=str(ROUTE_FILE))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--years", default="2030,2040,2050")
    parser.add_argument(
        "--policies",
        default="reference_policy,current_policy,strict_policy,critical_route_policy",
    )
    parser.add_argument("--solver-methods", default="highs,highs-ds,highs-ipm")
    parser.add_argument(
        "--elasticity-source-dirs",
        default="root,aggregate_us_eu,mc_smoke_china_2050_strict_fallback,mc_smoke_china_2050_strict_v2,eu_country_split",
    )
    parser.add_argument(
        "--skip-accessible-li",
        action="store_true",
        help="Skip single-technology max-accessible Li optimization.",
    )
    args = parser.parse_args()
    for output in write_outputs(args):
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
