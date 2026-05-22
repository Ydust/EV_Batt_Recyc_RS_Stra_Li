import argparse
import os
from pathlib import Path
import time

import pandas as pd

import joint_policy_transport_technology_optimization as joint
import scenario_transport_paths as transport_module


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "Figure_data" / "joint_policy_technology" / "dynamic_scale_cost"
BASE_METHODS = ["Direct", "Hydro", "Pyro"]
PYROHYDRO_METHOD = "PyroHydro"


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


def parse_key_value_floats(value):
    parsed = {}
    if not value:
        return parsed
    for item in parse_csv(value):
        key, multiplier = item.split("=", 1)
        parsed[key.strip()] = float(multiplier)
    return parsed


def safe_to_csv(frame, path, **kwargs):
    path = Path(path)
    tmp_path = path.with_name(f"{path.name}.tmp")
    last_error = None
    for _ in range(5):
        try:
            frame.to_csv(tmp_path, **kwargs)
            os.replace(tmp_path, path)
            return
        except OSError as error:
            last_error = error
            time.sleep(1.0)
    raise last_error


def summarize(routes, year, policy, cost_mode, methods):
    real = routes[~routes["is_unprocessed"].astype(bool)].copy()
    rows = []
    total_li = float(real["recovered_lithium_t"].sum())
    total_cost = joint.route_modeled_cost(routes)
    for technology in methods:
        tech = real[real["technology"] == technology]
        recovered_li = float(tech["recovered_lithium_t"].sum())
        scrap = float(tech["scrap_t"].sum())
        rows.append(
            {
                "year": year,
                "policy_scenario": policy,
                "cost_mode": cost_mode,
                "technology": technology,
                "scrap_t": scrap,
                "recovered_lithium_t": recovered_li,
                "technology_share_pct": recovered_li / total_li * 100.0 if total_li > 0 else 0.0,
                "route_modeled_cost": total_cost,
                "scale_iteration": int(routes["scale_iteration"].max())
                if "scale_iteration" in routes.columns
                else 0,
            }
        )
    return rows


def solve_fixed(
    supply,
    destination_capacity,
    distance,
    country_meta,
    year,
    policy,
    recovery,
    li_content,
    emission,
    methods,
):
    method_costs = joint.build_method_costs(
        distance,
        destination_capacity,
        country_meta,
        methods,
        policy,
        year,
        "hazardous",
        "recovery",
        joint.DEFAULT_DELAY_COST_USD_PER_T_DAY,
    )
    method_components = joint.build_method_cost_components(
        distance,
        destination_capacity,
        country_meta,
        methods,
        policy,
        year,
        "hazardous",
        "recovery",
        joint.DEFAULT_DELAY_COST_USD_PER_T_DAY,
    )
    routes = joint.solve_joint_domestic_priority(
        supply,
        destination_capacity,
        method_costs,
        methods,
        method_components,
    )
    return joint.add_lithium_outputs(routes, year, recovery, li_content, emission)


def solve_dynamic(
    supply,
    destination_capacity,
    distance,
    country_meta,
    year,
    policy,
    recovery,
    li_content,
    emission,
    max_iterations,
    tolerance,
    relaxation,
    require_country_technology_curve,
    availability_threshold,
    direct_cost_multiplier,
    methods,
):
    method_cost_multipliers = (
        {"Direct": direct_cost_multiplier}
        if abs(direct_cost_multiplier - 1.0) > 1e-12
        else None
    )
    routes, _ = joint.solve_joint_dynamic_scale(
        supply,
        destination_capacity,
        distance,
        country_meta,
        methods,
        policy,
        year,
        "hazardous",
        "recovery",
        joint.DEFAULT_DELAY_COST_USD_PER_T_DAY,
        max_iterations=max_iterations,
        tolerance=tolerance,
        relaxation=relaxation,
        require_country_technology_curve=require_country_technology_curve,
        availability_threshold=availability_threshold,
        method_cost_multipliers=method_cost_multipliers,
    )
    return joint.add_lithium_outputs(routes, year, recovery, li_content, emission)


def run(
    years,
    policies,
    max_iterations,
    tolerance,
    relaxation,
    require_country_technology_curve,
    availability_threshold,
    direct_cost_multiplier,
    include_pyrohydro,
    pyrohydro_pyro_weight,
    pyrohydro_group_multipliers,
    pyrohydro_country_multipliers,
    lp_solver,
    output_dir,
    skip_fixed,
):
    joint.configure_lp_solver(lp_solver)
    methods = BASE_METHODS + ([PYROHYDRO_METHOD] if include_pyrohydro else [])
    if include_pyrohydro:
        joint.configure_pyrohydro(
            pyro_weight=pyrohydro_pyro_weight,
            group_multipliers=pyrohydro_group_multipliers,
            country_multipliers=pyrohydro_country_multipliers,
        )
    transport_module.POLICY_FILE = ROOT / "Figure_data" / "joint_policy_technology" / "waste_trade_policy_constraints_reference_relaxed.csv"
    _, capacity, producer_iso, country_meta = joint.load_inputs("high_collection")
    scrap_by_type = joint.load_scrap_by_type("high_collection")
    countries = pd.read_csv(ROOT / "all_countries.csv")
    distance = joint.load_distance_matrix()
    li_content = joint.load_li_content()
    recovery = joint.load_recovery_efficiency("baseline", years, methods)
    emission = joint.load_emission_factor()

    output_dir.mkdir(parents=True, exist_ok=True)
    route_output = output_dir / "dynamic_scale_routes.csv"
    summary_output = output_dir / "dynamic_scale_summary.csv"
    existing_routes = pd.read_csv(route_output) if route_output.exists() else pd.DataFrame()
    existing_summary = pd.read_csv(summary_output) if summary_output.exists() else pd.DataFrame()
    completed = set()
    if not existing_summary.empty:
        completed = {
            (int(row["year"]), row["policy_scenario"])
            for _, row in existing_summary.iterrows()
            if row.get("technology") == "Direct"
            and row.get("cost_mode") != "fixed_median"
            and (
                ("pyrohydro" in str(row.get("cost_mode", ""))) == include_pyrohydro
            )
        }
    if not existing_routes.empty and completed:
        existing_routes = existing_routes[
            existing_routes.apply(
                lambda row: (
                    (int(row["year"]), row["policy_scenario"]) in completed
                    or row.get("cost_mode") == "fixed_median"
                ),
                axis=1,
            )
        ].copy()
    route_frames = [existing_routes] if not existing_routes.empty else []
    summary_rows = existing_summary.to_dict("records") if not existing_summary.empty else []
    for year in years:
        supply = joint.make_supply_by_type(scrap_by_type, countries, year, "Strategy 3")
        destination_capacity = joint.make_capacity(capacity, producer_iso, year)
        for policy in policies:
            if (year, policy) in completed:
                print(f"Skipping completed year={year}, policy={policy}", flush=True)
                continue
            if not skip_fixed:
                fixed_routes = solve_fixed(
                    supply,
                    destination_capacity,
                    distance,
                    country_meta,
                    year,
                    policy,
                    recovery,
                    li_content,
                    emission,
                    methods,
                )
                fixed_routes["year"] = year
                fixed_routes["policy_scenario"] = policy
                fixed_routes["cost_mode"] = "fixed_median"
                route_frames.append(fixed_routes)
                summary_rows.extend(
                    summarize(fixed_routes, year, policy, "fixed_median", methods)
                )

            dynamic_routes = solve_dynamic(
                supply,
                destination_capacity,
                distance,
                country_meta,
                year,
                policy,
                recovery,
                li_content,
                emission,
                max_iterations,
                tolerance,
                relaxation,
                require_country_technology_curve,
                availability_threshold,
                direct_cost_multiplier,
                methods,
            )
            dynamic_routes["year"] = year
            dynamic_routes["policy_scenario"] = policy
            cost_mode = (
                f"dynamic_scale_direct_x{direct_cost_multiplier:g}"
                if abs(direct_cost_multiplier - 1.0) > 1e-12
                else "dynamic_scale"
            )
            if availability_threshold is not None:
                cost_mode = f"{cost_mode}_avail_gte_{availability_threshold:g}"
            if include_pyrohydro:
                cost_mode = f"{cost_mode}_pyrohydro"
            dynamic_routes["cost_mode"] = cost_mode
            route_frames.append(dynamic_routes)
            summary_rows.extend(summarize(dynamic_routes, year, policy, cost_mode, methods))
            safe_to_csv(pd.concat(route_frames, ignore_index=True), route_output, index=False)
            safe_to_csv(pd.DataFrame(summary_rows), summary_output, index=False)
            print(f"Finished year={year}, policy={policy}", flush=True)
    return route_output, summary_output


def main():
    parser = argparse.ArgumentParser(
        description="Compare fixed median recycling cost with dynamic country-technology scale cost."
    )
    parser.add_argument("--years", default="2030,2040,2050")
    parser.add_argument(
        "--policies",
        default="reference_policy,current_policy,strict_policy,critical_route_policy",
    )
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--relaxation", type=float, default=0.5)
    parser.add_argument("--availability-threshold", type=float, default=None)
    parser.add_argument("--direct-cost-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--include-pyrohydro",
        action="store_true",
        help="Add PyroHydro as a fourth technology choice; default preserves the legacy three-technology model.",
    )
    parser.add_argument(
        "--pyrohydro-pyro-weight",
        type=float,
        default=None,
        help="Pyro share in PyroHydro cost; Hydro share is 1 minus this value.",
    )
    parser.add_argument(
        "--pyrohydro-group-multipliers",
        default="",
        help="Comma-separated group multipliers, e.g. developed=0.78,ev_producer=0.82,other=0.90.",
    )
    parser.add_argument(
        "--pyrohydro-country-multipliers",
        default="",
        help="Comma-separated country multipliers, e.g. USA=0.82,CHN=0.85.",
    )
    parser.add_argument("--lp-solver", choices=["highs", "gurobi"], default="highs")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--skip-fixed", action="store_true")
    parser.add_argument(
        "--allow-missing-country-technology-cost",
        action="store_true",
        help="Allow fallback costs when a country-technology cost curve is missing.",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if args.include_pyrohydro and output_dir == OUTPUT_DIR:
        output_dir = output_dir.with_name(f"{output_dir.name}_pyrohydro")
    for output in run(
        parse_years(args.years),
        parse_csv(args.policies),
        args.max_iterations,
        args.tolerance,
        args.relaxation,
        not args.allow_missing_country_technology_cost,
        args.availability_threshold,
        args.direct_cost_multiplier,
        args.include_pyrohydro,
        args.pyrohydro_pyro_weight,
        parse_key_value_floats(args.pyrohydro_group_multipliers),
        parse_key_value_floats(args.pyrohydro_country_multipliers),
        args.lp_solver,
        output_dir,
        args.skip_fixed,
    ):
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
