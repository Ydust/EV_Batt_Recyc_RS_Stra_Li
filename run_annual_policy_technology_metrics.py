import argparse
from pathlib import Path

import pandas as pd

import analyze_direct_entry_cost as direct_metrics
import analyze_policy_technology_mechanisms as mechanisms
import analyze_temporal_policy_technology as temporal
import joint_policy_transport_technology_optimization as joint_opt
import run_policy_objective_technology_elasticity as elasticity
import scenario_transport_paths as transport_module


ROOT = Path(__file__).resolve().parent
BASE_OUTPUT = ROOT / "Figure_data" / "joint_policy_technology"
POLICIES = ["reference_policy", "current_policy", "strict_policy", "critical_route_policy"]
BASE_METHODS = ["Direct", "Hydro", "Pyro"]
PYROHYDRO_METHOD = "PyroHydro"
TARGETS = ["China", "United States", "European Union"]
TARGET_SHARES = [0.0, 0.75, 0.9, 0.95, 0.99, 1.0]
YEARS = list(range(2030, 2051))


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_years(value):
    years = []
    for item in parse_csv(value):
        if "-" in item:
            start, end = item.split("-", 1)
            years.extend(range(int(start), int(end) + 1))
        else:
            years.append(int(item))
    return sorted(dict.fromkeys(years))


def output_name(years):
    return f"annual_{min(years)}_{max(years)}"


def route_modeled_cost(routes):
    real = routes[~routes["is_unprocessed"].astype(bool)].copy()
    return float(real[["transport_cost", "recycling_cost", "carbon_cost", "policy_cost"]].sum(axis=1).sum())


def run_annual_routes(years, policies, output_suffix, include_pyrohydro):
    methods = BASE_METHODS + ([PYROHYDRO_METHOD] if include_pyrohydro else [])
    transport_module.POLICY_FILE = joint_opt.ROOT / "waste_trade_policy_constraints_with_critical_routes.csv"
    _, capacity, producer_iso, country_meta = joint_opt.load_inputs("high_collection")
    scrap_by_type = joint_opt.load_scrap_by_type("high_collection")
    countries = pd.read_csv(ROOT / "all_countries.csv")
    distance = joint_opt.load_distance_matrix()
    li_content = joint_opt.load_li_content()
    recovery = joint_opt.load_recovery_efficiency("baseline", years, methods)
    emission = joint_opt.load_emission_factor()

    output_dir = BASE_OUTPUT / output_suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    route_output = output_dir / "joint_policy_transport_technology_routes.csv"
    summary_output = output_dir / "joint_policy_transport_technology_summary.csv"
    existing_routes = pd.read_csv(route_output) if route_output.exists() else pd.DataFrame()
    existing_summary = pd.read_csv(summary_output) if summary_output.exists() else pd.DataFrame()
    completed = set()
    if not existing_summary.empty:
        completed = {
            (int(row["year"]), row["policy_scenario"])
            for _, row in existing_summary.iterrows()
        }

    route_frames = [existing_routes] if not existing_routes.empty else []
    summary_rows = existing_summary.to_dict("records") if not existing_summary.empty else []
    for year in years:
        supply = joint_opt.make_supply_by_type(scrap_by_type, countries, year, "Strategy 3")
        destination_capacity = joint_opt.make_capacity(capacity, producer_iso, year)
        for policy in policies:
            if (year, policy) in completed:
                print(f"Skipping completed route year={year}, policy={policy}", flush=True)
                continue
            method_costs = joint_opt.build_method_costs(
                distance,
                destination_capacity,
                country_meta,
                methods,
                policy,
                year,
                "hazardous",
                "recovery",
                joint_opt.DEFAULT_DELAY_COST_USD_PER_T_DAY,
            )
            method_components = joint_opt.build_method_cost_components(
                distance,
                destination_capacity,
                country_meta,
                methods,
                policy,
                year,
                "hazardous",
                "recovery",
                joint_opt.DEFAULT_DELAY_COST_USD_PER_T_DAY,
            )
            routes = joint_opt.solve_joint_domestic_priority(
                supply,
                destination_capacity,
                method_costs,
                methods,
                method_components,
            )
            routes = joint_opt.add_lithium_outputs(routes, year, recovery, li_content, emission)
            routes["year"] = year
            routes["policy_scenario"] = policy
            routes["strategy"] = "Strategy 3"
            route_frames.append(routes)
            summary_rows.append(joint_opt.summarize(routes, year, policy, "Strategy 3"))
            pd.concat(route_frames, ignore_index=True).to_csv(route_output, index=False)
            pd.DataFrame(summary_rows).to_csv(summary_output, index=False)
            print(f"Finished route year={year}, policy={policy}", flush=True)
    return route_output, summary_output


def run_annual_direct_elasticity(years, policies, targets, output_suffix, solver_methods):
    elasticity.run(
        years=years,
        policies=policies,
        target_shares=TARGET_SHARES,
        target_mode="aggregate",
        target_names=targets,
        output_suffix=output_suffix,
        mc_draws=1,
        random_seed=20260507,
        cost_noise_sd=0.0,
        recovery_noise_sd=0.0,
        solver_methods=solver_methods,
    )
    summary, mix = direct_metrics.read_source_tables([output_suffix])
    summary, mix = direct_metrics.normalize_tables(summary, mix)
    output_dir = BASE_OUTPUT / "direct_entry_cost" / output_suffix
    direct_metrics.write_outputs(
        summary,
        mix,
        output_dir,
        direct_metrics.ENTRY_THRESHOLDS_PCT,
    )
    return output_dir


def run_mechanisms(years, policies, route_file, output_suffix, skip_accessible_li, solver_methods):
    args = argparse.Namespace(
        route_file=str(route_file),
        output_dir=str(BASE_OUTPUT / "policy_technology_mechanisms" / output_suffix),
        years=",".join(map(str, years)),
        policies=",".join(policies),
        solver_methods=",".join(solver_methods),
        elasticity_source_dirs=output_suffix,
        skip_accessible_li=skip_accessible_li,
    )
    mechanisms.write_outputs(args)
    return Path(args.output_dir)


def run_temporal(direct_dir, mechanism_dir, output_suffix):
    output_dir = BASE_OUTPUT / "temporal_policy_technology" / output_suffix
    args = argparse.Namespace(
        direct_dir=str(direct_dir),
        mechanism_dir=str(mechanism_dir),
        output_dir=str(output_dir),
    )
    # Reuse the temporal module's functions directly to avoid shelling out.
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "direct_entry_threshold_temporal.csv": temporal.direct_threshold_temporal(Path(args.direct_dir)),
        "cost_95_to_100_temporal.csv": temporal.cost_temporal(Path(args.direct_dir)),
        "technology_accessible_li_temporal.csv": temporal.accessible_li_temporal(Path(args.mechanism_dir)),
        "route_reallocation_temporal.csv": temporal.route_temporal(Path(args.mechanism_dir)),
        "destination_shift_temporal.csv": temporal.destination_temporal(Path(args.mechanism_dir)),
    }
    outputs["policy_time_summary.csv"] = temporal.policy_time_summary(
        outputs["cost_95_to_100_temporal.csv"],
        outputs["route_reallocation_temporal.csv"],
    )
    for name, data in outputs.items():
        data.to_csv(output_dir / name, index=False)
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run annual policy-technology indicators without overwriting node-year outputs."
    )
    parser.add_argument("--years", default="2030-2050")
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--solver-methods", default="highs,highs-ds,highs-ipm")
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--skip-routes", action="store_true")
    parser.add_argument("--skip-direct", action="store_true")
    parser.add_argument("--skip-accessible-li", action="store_true")
    parser.add_argument(
        "--include-pyrohydro",
        action="store_true",
        help="Add PyroHydro as a fourth route technology; default preserves the legacy three-technology model.",
    )
    parser.add_argument(
        "--direct-only",
        action="store_true",
        help="Run only annual Direct threshold/cost elasticity and its post-processing.",
    )
    parser.add_argument(
        "--routes-only",
        action="store_true",
        help="Run only annual economic route optimization and route mechanism post-processing.",
    )
    args = parser.parse_args()
    years = parse_years(args.years)
    policies = parse_csv(args.policies)
    targets = parse_csv(args.targets)
    solver_methods = parse_csv(args.solver_methods)
    output_suffix = args.output_suffix or output_name(years)
    if args.include_pyrohydro and not args.output_suffix:
        output_suffix = f"{output_suffix}_pyrohydro"

    route_file = BASE_OUTPUT / output_suffix / "joint_policy_transport_technology_routes.csv"
    direct_dir = BASE_OUTPUT / "direct_entry_cost" / output_suffix
    mechanism_dir = BASE_OUTPUT / "policy_technology_mechanisms" / output_suffix

    if not args.skip_routes and not args.direct_only:
        route_file, _ = run_annual_routes(
            years, policies, output_suffix, args.include_pyrohydro
        )
    if not args.skip_direct and not args.routes_only:
        direct_dir = run_annual_direct_elasticity(
            years,
            policies,
            targets,
            output_suffix,
            solver_methods,
        )
    if not args.direct_only:
        mechanism_dir = run_mechanisms(
            years,
            policies,
            route_file,
            output_suffix,
            args.skip_accessible_li,
            solver_methods,
        )
    if direct_dir.exists() and mechanism_dir.exists():
        temporal_dir = run_temporal(direct_dir, mechanism_dir, output_suffix)
        print(f"Wrote annual temporal outputs to {temporal_dir}")
    print("Annual run complete.")


if __name__ == "__main__":
    main()
