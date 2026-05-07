import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE_POLICY_FILE = ROOT / "waste_trade_policy_constraints.csv"
USED_ROUTES_FILE = (
    ROOT
    / "Figure_data"
    / "policy_constraint_strength"
    / "policy_constraint_strength_used_routes.csv"
)
OUTPUT_POLICY_FILE = ROOT / "waste_trade_policy_constraints_with_critical_routes.csv"


def select_critical_routes(used_routes, base_policy, method, strategy, top_n, years):
    routes = used_routes.copy()
    routes = routes[
        (routes["method"] == method)
        & (routes["strategy"] == strategy)
        & (routes["policy_scenario"] == base_policy)
        & (routes["cross_border"] == True)
    ].copy()
    routes = routes[
        ~routes["source_iso3"].astype(str).str.startswith("Virtual")
        & ~routes["destination_iso3"].astype(str).str.startswith("Virtual")
    ].copy()
    if years:
        routes = routes[routes["year"].isin(years)].copy()
    if routes.empty:
        raise ValueError("No used cross-border routes found for critical-route selection.")

    selected = (
        routes.groupby(["source_iso3", "destination_iso3"], as_index=False)["scrap_t"]
        .sum()
        .sort_values("scrap_t", ascending=False)
        .head(top_n)
    )
    return selected


def country_lookup():
    countries = pd.read_csv(ROOT / "nation_list_new.csv")[["region", "iso3"]]
    return dict(zip(countries["iso3"], countries["region"]))


def append_critical_policy(
    critical_routes,
    base_policy,
    scenario_name,
    start_year,
    end_year,
    source_basis,
    output_file,
):
    policy = pd.read_csv(BASE_POLICY_FILE)
    inherited = policy[policy["scenario"] == base_policy].copy()
    inherited["scenario"] = scenario_name
    iso_to_country = country_lookup()
    new_rows = []
    for _, route in critical_routes.iterrows():
        source_country = iso_to_country.get(route["source_iso3"])
        destination_country = iso_to_country.get(route["destination_iso3"])
        if not source_country or not destination_country:
            continue
        new_rows.append(
            {
                "scenario": scenario_name,
                "source_group": "",
                "destination_group": "",
                "source_country": source_country,
                "destination_country": destination_country,
                "start_year": start_year,
                "end_year": end_year,
                "waste_class": "hazardous",
                "treatment_type": "recovery",
                "forbidden": 1,
                "policy_penalty_usd_per_t": 0,
                "approval_delay_days": 0,
                "source_basis": source_basis,
                "notes": (
                    f"Critical-route stress test: route selected from observed optimized "
                    f"cross-border flows, cumulative scrap_t={route['scrap_t']:.4f}."
                ),
            }
        )

    output = pd.concat([policy, inherited, pd.DataFrame(new_rows)], ignore_index=True)
    output.to_csv(output_file, index=False)
    return output_file, pd.DataFrame(new_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a critical-route strict policy scenario from actually used optimized routes."
    )
    parser.add_argument("--used-routes", default=str(USED_ROUTES_FILE))
    parser.add_argument("--base-policy", default="reference_policy")
    parser.add_argument("--scenario-name", default="critical_route_policy")
    parser.add_argument("--method", default="Direct")
    parser.add_argument("--strategy", default="Strategy 3")
    parser.add_argument("--top-n", type=int, default=12)
    parser.add_argument("--years", default="2045,2050")
    parser.add_argument("--start-year", type=int, default=2025)
    parser.add_argument("--end-year", type=int, default=2050)
    parser.add_argument(
        "--source-basis", default="critical_route_stress_test"
    )
    parser.add_argument("--output", default=str(OUTPUT_POLICY_FILE))
    args = parser.parse_args()

    used_routes = pd.read_csv(args.used_routes)
    years = [int(item.strip()) for item in args.years.split(",") if item.strip()]
    critical_routes = select_critical_routes(
        used_routes,
        args.base_policy,
        args.method,
        args.strategy,
        args.top_n,
        years,
    )
    output, new_rows = append_critical_policy(
        critical_routes,
        args.base_policy,
        args.scenario_name,
        args.start_year,
        args.end_year,
        args.source_basis,
        Path(args.output),
    )

    critical_path = Path(args.output).with_name("critical_route_policy_selected_routes.csv")
    critical_routes.to_csv(critical_path, index=False)
    print(f"Wrote {output}")
    print(f"Wrote {critical_path}")
    print(f"Added {len(new_rows)} critical-route policy rows")


if __name__ == "__main__":
    main()
