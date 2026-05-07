import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
ROUTES_FILE = (
    ROOT
    / "Figure_data"
    / "policy_constraint_strength"
    / "policy_constraint_strength_used_routes.csv"
)
COUNTRY_FILE = ROOT / "nation_list_new.csv"
OUTPUT_DIR = ROOT / "Figure_data" / "policy_constraint_strength"


def parse_years(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def country_lookup():
    countries = pd.read_csv(COUNTRY_FILE)[["iso3", "region"]].dropna()
    return dict(zip(countries["iso3"], countries["region"]))


def summarize_changes(
    routes,
    years,
    method,
    strategy,
    baseline_policy,
    comparison_policy,
):
    data = routes[
        (routes["year"].isin(years))
        & (routes["method"] == method)
        & (routes["strategy"] == strategy)
        & (routes["cross_border"] == True)
        & (~routes["source_iso3"].astype(str).str.startswith("Virtual"))
        & (~routes["destination_iso3"].astype(str).str.startswith("Virtual"))
        & (routes["policy_scenario"].isin([baseline_policy, comparison_policy]))
    ].copy()

    grouped = (
        data.groupby(
            ["year", "policy_scenario", "source_iso3", "destination_iso3"],
            as_index=False,
        )["scrap_t"]
        .sum()
        .pivot_table(
            index=["year", "source_iso3", "destination_iso3"],
            columns="policy_scenario",
            values="scrap_t",
            fill_value=0,
        )
        .reset_index()
    )
    for policy in [baseline_policy, comparison_policy]:
        if policy not in grouped.columns:
            grouped[policy] = 0.0
    grouped["delta_scrap_t"] = grouped[comparison_policy] - grouped[baseline_policy]
    grouped = grouped[grouped["delta_scrap_t"].abs() > 1e-9].copy()

    lookup = country_lookup()
    grouped["source_country"] = grouped["source_iso3"].map(lookup).fillna(
        grouped["source_iso3"]
    )
    grouped["destination_country"] = grouped["destination_iso3"].map(lookup).fillna(
        grouped["destination_iso3"]
    )
    grouped["route"] = grouped["source_country"] + " -> " + grouped["destination_country"]
    grouped["change_type"] = grouped["delta_scrap_t"].where(
        grouped["delta_scrap_t"] > 0, other=0
    )
    grouped["change_type"] = grouped["delta_scrap_t"].apply(
        lambda value: "new_or_increased" if value > 0 else "reduced_or_removed"
    )
    return grouped.sort_values(["year", "delta_scrap_t"], ascending=[True, True])


def main():
    parser = argparse.ArgumentParser(
        description="Summarize country-pair route changes between two policy scenarios."
    )
    parser.add_argument("--years", default="2045,2050")
    parser.add_argument("--method", default="Direct")
    parser.add_argument("--strategy", default="Strategy 3")
    parser.add_argument("--baseline-policy", default="reference_policy")
    parser.add_argument("--comparison-policy", default="critical_route_policy")
    args = parser.parse_args()

    routes = pd.read_csv(ROUTES_FILE)
    changes = summarize_changes(
        routes,
        parse_years(args.years),
        args.method,
        args.strategy,
        args.baseline_policy,
        args.comparison_policy,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = OUTPUT_DIR / "policy_route_changes_country_pairs.csv"
    changes.to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
