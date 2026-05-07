import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
POLICY_FILE = ROOT / "battery_policy_alignment.csv"
COUNTRY_FILE = ROOT / "nation_list_new.csv"
OUTPUT_DIR = ROOT / "Scenario result" / "policy_alignment"

EU_COUNTRIES = {
    "Austria",
    "Belgium",
    "Bulgaria",
    "Croatia",
    "Cyprus",
    "Czechia",
    "Denmark",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Ireland",
    "Italy",
    "Latvia",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Netherlands",
    "Poland",
    "Portugal",
    "Romania",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Sweden",
}


OBJECT_SCOPES = ["ev_power_battery", "stationary_storage_battery"]
MAX_FIELDS = [
    "collection_rate_floor",
    "li_recovery_efficiency_floor",
    "recycled_content_li_min",
    "traceability_requirement",
    "battery_passport_requirement",
    "epr_strength",
    "domestic_content_pressure",
    "hazardous_transport_penalty",
]
SUM_FIELDS = ["storage_deployment_target_gwh"]


def parse_years(value):
    if value:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return list(range(2025, 2051))


def load_countries():
    countries = pd.read_csv(COUNTRY_FILE)[["region", "iso3", "continent"]].drop_duplicates()
    countries = countries.rename(columns={"region": "country"})
    return countries


def countries_for_policy(rule, countries):
    explicit_country = rule.get("country")
    if pd.notna(explicit_country) and str(explicit_country).strip():
        return countries[countries["country"] == str(explicit_country).strip()].copy()

    region = str(rule.get("policy_region", "")).strip()
    if region == "EU":
        return countries[countries["country"].isin(EU_COUNTRIES)].copy()
    if region == "ALL":
        return countries.copy()
    return countries[countries["country"] == region].copy()


def object_scopes_for_rule(rule):
    scope = str(rule["object_scope"]).strip()
    if scope == "all_battery":
        return OBJECT_SCOPES
    return [scope]


def expand_policy_rules(years):
    policies = pd.read_csv(POLICY_FILE)
    countries = load_countries()
    rows = []
    for _, rule in policies.iterrows():
        target_countries = countries_for_policy(rule, countries)
        if target_countries.empty:
            continue
        rule_years = [
            year
            for year in years
            if int(rule["start_year"]) <= year <= int(rule["end_year"])
        ]
        if not rule_years:
            continue
        for _, country in target_countries.iterrows():
            for year in rule_years:
                for object_scope in object_scopes_for_rule(rule):
                    row = rule.to_dict()
                    row.update(
                        {
                            "Year": year,
                            "country": country["country"],
                            "iso3": country["iso3"],
                            "continent": country["continent"],
                            "effective_object_scope": object_scope,
                        }
                    )
                    rows.append(row)
    return pd.DataFrame(rows)


def summarize_policy(expanded):
    if expanded.empty:
        return pd.DataFrame()
    for field in MAX_FIELDS + SUM_FIELDS + ["ev_sales_target"]:
        expanded[field] = pd.to_numeric(expanded[field], errors="coerce")

    grouped = []
    for keys, group in expanded.groupby(["Year", "country", "iso3", "effective_object_scope"]):
        row = {
            "Year": keys[0],
            "country": keys[1],
            "iso3": keys[2],
            "object_scope": keys[3],
            "policy_count": len(group),
            "active_policy_ids": ";".join(group["policy_id"].astype(str).unique()),
            "source_basis": ";".join(group["source_basis"].astype(str).unique()),
        }
        for field in MAX_FIELDS:
            value = group[field].max(skipna=True)
            row[field] = 0.0 if pd.isna(value) else float(value)
        for field in SUM_FIELDS:
            value = group[field].sum(skipna=True)
            row[field] = 0.0 if pd.isna(value) else float(value)

        ev_targets = group["ev_sales_target"].dropna()
        if keys[3] == "ev_power_battery" and not ev_targets.empty:
            row["ev_sales_target"] = float(ev_targets.max())
        else:
            row["ev_sales_target"] = 0.0

        row["policy_collection_multiplier"] = 1.0 + 0.20 * row["epr_strength"] + 0.10 * row["traceability_requirement"]
        row["policy_recovery_multiplier"] = 1.0 + 0.15 * min(row["li_recovery_efficiency_floor"], 1.0)
        row["secondary_li_demand_pull"] = (
            row["recycled_content_li_min"]
            + 0.05 * row["domestic_content_pressure"]
            + 0.03 * row["battery_passport_requirement"]
        )
        row["trade_policy_penalty_multiplier"] = 1.0 + row["hazardous_transport_penalty"]
        grouped.append(row)

    return pd.DataFrame(grouped).sort_values(["Year", "country", "object_scope"])


def main():
    parser = argparse.ArgumentParser(
        description="Expand EV and stationary-storage battery policies into country-year model parameters."
    )
    parser.add_argument("--years", default="2025,2030,2035,2040,2045,2050")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    years = parse_years(args.years)
    expanded = expand_policy_rules(years)
    expanded_path = OUTPUT_DIR / "battery_policy_alignment_expanded.csv"
    expanded.to_csv(expanded_path, index=False)

    summary = summarize_policy(expanded)
    summary_path = OUTPUT_DIR / "battery_policy_country_year_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {expanded_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
