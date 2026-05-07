import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import run_lithium_loss_scenarios as scenarios
import scenario_transport_paths as transport_module
from joint_policy_transport_technology_optimization import build_method_cost_components, build_method_costs
from scenario_transport_paths import DEFAULT_DELAY_COST_USD_PER_T_DAY, load_distance_matrix, load_inputs


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "Figure_data" / "joint_policy_technology" / "policy_objective_technology_elasticity"
YEARS = [2030, 2040, 2050]
POLICIES = ["reference_policy", "current_policy", "strict_policy", "critical_route_policy"]
TARGET_SHARES = [0.0, 0.75, 0.9, 0.95, 0.99, 1.0]
METHODS = ["Direct", "Hydro", "Pyro"]
STRATEGY = "Strategy 3"
WASTE_CLASS = "hazardous"
TREATMENT_TYPE = "recovery"
CAPACITY_EXPANSION_COUNTRIES = ["CHN", "KOR", "JPN", "USA", "IND"]
RECOVERY_FLOOR_BY_METHOD = {"Direct": 0.97, "Hydro": 0.95, "Pyro": 0.35}


EU_COUNTRIES = {
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
    "Denmark", "Estonia", "Finland", "France", "Germany", "Greece",
    "Hungary", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg",
    "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia",
    "Slovenia", "Spain", "Sweden",
}


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_years(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_csv(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def route_modeled_cost(routes):
    real = routes[~routes["is_unprocessed"].astype(bool)].copy()
    return float(real[["transport_cost", "recycling_cost", "carbon_cost", "policy_cost"]].sum(axis=1).sum())


def load_target_regions(target_mode, target_names=None):
    countries = pd.read_csv(ROOT / "all_countries.csv").dropna(subset=["country", "iso3"])
    country_to_iso = countries.drop_duplicates("country").set_index("country")["iso3"].to_dict()
    targets = {
        "China": {"CHN"},
        "United States": {"USA"},
        "European Union": set(countries.loc[countries["country"].isin(EU_COUNTRIES), "iso3"]),
    }
    if target_mode == "eu-countries":
        targets = {
            country: {country_to_iso[country]}
            for country in sorted(EU_COUNTRIES)
            if country in country_to_iso
        }
    if target_names:
        targets = {name: targets[name] for name in target_names if name in targets}
    return targets


def summarize(
    routes,
    year,
    policy,
    target_region,
    target_iso,
    target_share,
    max_target_li,
    solver_target_li,
):
    real = routes[~routes["is_unprocessed"].astype(bool)].copy()
    target = real[real["destination_iso3"].isin(target_iso)].copy()
    global_li = float(real["recovered_lithium_t"].sum())
    target_li = float(target["recovered_lithium_t"].sum())
    return {
        "year": year,
        "policy_scenario": policy,
        "target_region": target_region,
        "target_share_of_max": target_share,
        "target_constraint_li_t": max_target_li * target_share,
        "solver_target_constraint_li_t": solver_target_li,
        "max_target_recovered_lithium_t": max_target_li,
        "global_recovered_lithium_t": global_li,
        "target_recovered_lithium_t": target_li,
        "target_max_attainment_pct": target_li / max_target_li * 100.0 if max_target_li > 0 else 0.0,
        "route_modeled_cost": route_modeled_cost(routes),
        "processed_scrap_t": float(real["scrap_t"].sum()),
        "solver_method": (
            str(routes["solver_method"].dropna().iloc[0])
            if "solver_method" in routes.columns and not routes["solver_method"].dropna().empty
            else ""
        ),
    }


def mix_rows(routes, year, policy, target_region, target_iso, target_share):
    real = routes[~routes["is_unprocessed"].astype(bool)].copy()
    target = real[real["destination_iso3"].isin(target_iso)].copy()
    rows = []
    for scope, data in [("global", real), ("target_region", target)]:
        total = float(data["recovered_lithium_t"].sum())
        by_tech = data.groupby("technology")["recovered_lithium_t"].sum()
        for technology in METHODS:
            value = float(by_tech.get(technology, 0.0))
            rows.append(
                {
                    "year": year,
                    "policy_scenario": policy,
                    "target_region": target_region,
                    "target_share_of_max": target_share,
                    "scope": scope,
                    "technology": technology,
                    "recovered_lithium_t": value,
                    "technology_share_pct": value / total * 100.0 if total > 0 else 0.0,
                }
            )
    return rows


def apply_technology_cost_noise(method_costs, method_components, multipliers):
    for method, multiplier in multipliers.items():
        if method not in method_costs or abs(multiplier - 1.0) < 1e-12:
            continue
        recycling = method_components[method]["recycling"]
        delta = recycling * (multiplier - 1.0)
        method_costs[method] = method_costs[method].add(delta, axis="columns")
        method_components[method]["recycling"] = recycling * multiplier
        method_components[method]["total"] = method_components[method]["total"].add(
            delta, axis="columns"
        )


def apply_recovery_noise(recovery, years, rng, noise_sd):
    if noise_sd <= 0:
        return recovery
    adjusted = dict(recovery)
    for year in years:
        for method in METHODS:
            key = (year, method)
            value = float(adjusted.get(key, 0.0))
            if value <= 0:
                continue
            noisy = value * float(np.exp(rng.normal(0.0, noise_sd)))
            adjusted[key] = min(max(noisy, 0.0), 0.995)
    return adjusted


def draw_cost_multipliers(rng, noise_sd):
    if noise_sd <= 0:
        return {method: 1.0 for method in METHODS}
    return {
        method: float(np.exp(rng.normal(0.0, noise_sd)))
        for method in METHODS
    }


def run(
    years,
    policies,
    target_shares,
    target_mode,
    target_names,
    output_suffix,
    mc_draws,
    random_seed,
    cost_noise_sd,
    recovery_noise_sd,
    solver_methods,
):
    transport_module.POLICY_FILE = scenarios.POLICY_FILE
    _, capacity, producer_iso, country_meta = load_inputs("high_collection")
    scrap_by_type = scenarios.load_scrap_by_type("high_collection")
    countries = pd.read_csv(ROOT / "all_countries.csv")
    distance = load_distance_matrix()
    li_content = scenarios.load_li_content()
    base_recovery = scenarios.load_recovery_efficiency("baseline", years, METHODS)
    base_recovery = scenarios.apply_recovery_floor_by_method(
        base_recovery, years, RECOVERY_FLOOR_BY_METHOD
    )
    emission = scenarios.load_emission_factor()
    lithium_price = scenarios.load_lithium_price("baseline")
    target_regions = load_target_regions(target_mode, target_names)
    run_config = {
        "years": ",".join(map(str, years)),
        "policies": ",".join(policies),
        "target_shares": ",".join(map(str, target_shares)),
        "target_mode": target_mode,
        "target_names": ",".join(target_names or []),
        "mc_draws": mc_draws,
        "random_seed": random_seed,
        "cost_noise_sd": cost_noise_sd,
        "recovery_noise_sd": recovery_noise_sd,
        "solver_methods": ",".join(solver_methods),
    }

    output_dir = OUTPUT_DIR / output_suffix if output_suffix else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_output = output_dir / "policy_objective_technology_elasticity_summary.csv"
    mix_output = output_dir / "policy_objective_technology_elasticity_mix.csv"
    config_output = output_dir / "policy_objective_technology_elasticity_config.csv"
    pd.DataFrame([run_config]).to_csv(config_output, index=False)
    summary_rows = (
        pd.read_csv(summary_output).to_dict("records") if summary_output.exists() else []
    )
    all_mix_rows = pd.read_csv(mix_output).to_dict("records") if mix_output.exists() else []
    completed = {
        (
            int(row.get("draw_id", 0)),
            int(row["year"]),
            row["policy_scenario"],
            row["target_region"],
            round(float(row["target_share_of_max"]), 8),
        )
        for row in summary_rows
    }

    for draw_id in range(mc_draws):
        rng = np.random.default_rng(random_seed + draw_id)
        recovery = apply_recovery_noise(base_recovery, years, rng, recovery_noise_sd)
        cost_multipliers_by_context = {}
        for year in years:
            for policy in policies:
                cost_multipliers_by_context[(year, policy)] = draw_cost_multipliers(
                    rng, cost_noise_sd
                )

        for year in years:
            supply = scenarios.make_supply_by_type(scrap_by_type, countries, year, STRATEGY)
            destination_capacity = scenarios.make_capacity(capacity, producer_iso, year)
            destination_capacity = scenarios.expand_key_capacity(destination_capacity, CAPACITY_EXPANSION_COUNTRIES, 1.25)

            for policy in policies:
                method_costs = build_method_costs(
                    distance, destination_capacity, country_meta, METHODS, policy, year,
                    WASTE_CLASS, TREATMENT_TYPE, DEFAULT_DELAY_COST_USD_PER_T_DAY
                )
                method_components = build_method_cost_components(
                    distance, destination_capacity, country_meta, METHODS, policy, year,
                    WASTE_CLASS, TREATMENT_TYPE, DEFAULT_DELAY_COST_USD_PER_T_DAY
                )
                cost_multipliers = cost_multipliers_by_context[(year, policy)]
                apply_technology_cost_noise(method_costs, method_components, cost_multipliers)

                for target_region, target_iso in target_regions.items():
                    max_routes = scenarios.solve_joint_scenario(
                        supply, destination_capacity, method_costs, METHODS, method_components,
                        year, li_content, recovery, lithium_price, objective="max_target_lithium",
                        target_destinations=target_iso,
                        solver_methods=solver_methods,
                    )
                    max_routes = scenarios.add_lithium_outputs(max_routes, year, recovery, li_content, emission)
                    max_target_li = float(max_routes[max_routes["destination_iso3"].isin(target_iso)]["recovered_lithium_t"].sum())

                    for target_share in target_shares:
                        key = (
                            draw_id,
                            year,
                            policy,
                            target_region,
                            round(float(target_share), 8),
                        )
                        if key in completed:
                            print(
                                f"Skipping completed draw={draw_id}, year={year}, policy={policy}, "
                                f"target={target_region}, share={target_share:.2f}",
                                flush=True,
                            )
                            continue
                        solver_target_li = max_target_li * target_share
                        if solver_target_li > 0:
                            solver_target_li *= 1.0 - 1e-6
                        try:
                            routes = scenarios.solve_joint_scenario(
                                supply,
                                destination_capacity,
                                method_costs,
                                METHODS,
                                method_components,
                                year,
                                li_content,
                                recovery,
                                lithium_price,
                                objective="cost_min",
                                min_target_recovered_lithium_t=solver_target_li,
                                target_destinations=target_iso,
                                solver_methods=solver_methods,
                            )
                        except RuntimeError as exc:
                            row = {
                                "draw_id": draw_id,
                                "random_seed": random_seed + draw_id,
                                "year": year,
                                "policy_scenario": policy,
                                "target_region": target_region,
                                "target_share_of_max": target_share,
                                "target_constraint_li_t": max_target_li * target_share,
                                "solver_target_constraint_li_t": solver_target_li,
                                "max_target_recovered_lithium_t": max_target_li,
                                "solve_status": "failed",
                                "solve_message": str(exc),
                            }
                            for method, multiplier in cost_multipliers.items():
                                row[f"{method.lower()}_cost_multiplier"] = multiplier
                                row[f"{method.lower()}_recovery_efficiency"] = recovery.get(
                                    (year, method), 0.0
                                )
                            summary_rows.append(row)
                            pd.DataFrame(summary_rows).to_csv(summary_output, index=False)
                            print(
                                f"Failed draw={draw_id}, year={year}, policy={policy}, "
                                f"target={target_region}, share={target_share:.2f}: {exc}",
                                flush=True,
                            )
                            continue
                        routes = scenarios.add_lithium_outputs(routes, year, recovery, li_content, emission)
                        routes["year"] = year
                        routes["policy_scenario"] = policy
                        routes["target_region"] = target_region
                        routes["target_share_of_max"] = target_share
                        row = summarize(
                            routes,
                            year,
                            policy,
                            target_region,
                            target_iso,
                            target_share,
                            max_target_li,
                            solver_target_li,
                        )
                        row["draw_id"] = draw_id
                        row["random_seed"] = random_seed + draw_id
                        row["solve_status"] = "success"
                        row["solve_message"] = ""
                        for method, multiplier in cost_multipliers.items():
                            row[f"{method.lower()}_cost_multiplier"] = multiplier
                            row[f"{method.lower()}_recovery_efficiency"] = recovery.get(
                                (year, method), 0.0
                            )
                        summary_rows.append(row)
                        rows = mix_rows(routes, year, policy, target_region, target_iso, target_share)
                        for mix_row in rows:
                            mix_row["draw_id"] = draw_id
                            mix_row["random_seed"] = random_seed + draw_id
                        all_mix_rows.extend(rows)
                        pd.DataFrame(summary_rows).to_csv(summary_output, index=False)
                        pd.DataFrame(all_mix_rows).to_csv(mix_output, index=False)
                        print(
                            f"Finished draw={draw_id}, year={year}, policy={policy}, "
                            f"target={target_region}, share={target_share:.2f}",
                            flush=True,
                        )

    pd.DataFrame(summary_rows).to_csv(summary_output, index=False)
    pd.DataFrame(all_mix_rows).to_csv(mix_output, index=False)
    return summary_output, mix_output


def main():
    parser = argparse.ArgumentParser(
        description="Run policy-constrained technology elasticity under soft regional Li-access targets."
    )
    parser.add_argument("--years", default="2030,2040,2050")
    parser.add_argument(
        "--policies",
        default="reference_policy,current_policy,strict_policy,critical_route_policy",
    )
    parser.add_argument("--target-shares", default="0,0.75,0.9,0.95,0.99,1")
    parser.add_argument(
        "--target-mode",
        choices=["aggregate", "eu-countries"],
        default="aggregate",
    )
    parser.add_argument(
        "--targets",
        default="",
        help="Optional comma-separated target names. For aggregate mode: China,United States,European Union.",
    )
    parser.add_argument(
        "--output-suffix",
        default="",
        help="Optional subfolder under the elasticity output directory.",
    )
    parser.add_argument("--mc-draws", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=20260507)
    parser.add_argument(
        "--cost-noise-sd",
        type=float,
        default=0.0,
        help="Lognormal sigma for technology recycling-cost multipliers.",
    )
    parser.add_argument(
        "--recovery-noise-sd",
        type=float,
        default=0.0,
        help="Lognormal sigma for technology Li-recovery efficiencies.",
    )
    parser.add_argument(
        "--solver-methods",
        default="gurobi,highs,highs-ds,highs-ipm",
        help="Comma-separated solver fallback order. Use gurobi only when gurobipy/license is available.",
    )
    args = parser.parse_args()
    target_names = parse_csv(args.targets) if args.targets else None
    for output in run(
        parse_years(args.years),
        parse_csv(args.policies),
        parse_float_csv(args.target_shares),
        args.target_mode,
        target_names,
        args.output_suffix,
        args.mc_draws,
        args.random_seed,
        args.cost_noise_sd,
        args.recovery_noise_sd,
        parse_csv(args.solver_methods),
    ):
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
