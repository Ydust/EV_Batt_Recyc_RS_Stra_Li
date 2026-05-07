import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SCRAP_FILE = ROOT / "Scenario result" / "EV_battery_inuse_scrap.csv"
METAL_CONTENT_FILE = ROOT / "cost" / "Metal content.csv"
RECOVERY_EFFICIENCY_FILE = ROOT / "cost" / "Recovery efficiency.csv"
RATE_FILE = ROOT / "country_recycling_rate_scenarios.csv"
TECH_RECOVERY_SCENARIO_FILE = ROOT / "technology_lithium_recovery_scenarios.csv"
COUNTRY_RATE_DRIVER_FILE = ROOT / "country_recycling_rate_drivers.csv"
TECH_RECOVERY_DRIVER_FILE = ROOT / "technology_recovery_efficiency_drivers.csv"
ALL_COUNTRIES_FILE = ROOT / "all_countries.csv"
DEVELOPED_NATION_FILE = ROOT / "developed_nation_list.csv"
OUTPUT_DIR = ROOT / "Scenario result" / "recycling_rate"


BATTERY_RECOVERY_ROWS = {
    "LFP": "LFP",
    "NMC111": "NMC(111)",
    "NMC523": "NMC(532)",
    "NMC622": "NMC(622)",
    "NMC811": "NMC(811)",
    "NCA": "NCA",
}


def _linear_rate(year, start_year, end_year, start_rate, end_rate):
    if end_year == start_year:
        return end_rate
    progress = (year - start_year) / (end_year - start_year)
    progress = min(max(progress, 0), 1)
    return start_rate + progress * (end_rate - start_rate)


def _logistic_value(year, start_value, max_value, midpoint, k):
    return start_value + (max_value - start_value) / (
        1 + np.exp(-k * (year - midpoint))
    )


def load_country_groups(scrap_df):
    countries = pd.DataFrame({"country": sorted(scrap_df["region"].unique())})
    countries["country_group"] = "other"

    if ALL_COUNTRIES_FILE.exists():
        all_countries = pd.read_csv(ALL_COUNTRIES_FILE)
        producer_countries = set(
            all_countries.loc[all_countries["producer"] == True, "country"]
        )
        countries.loc[countries["country"].isin(producer_countries), "country_group"] = (
            "ev_producer"
        )

    if DEVELOPED_NATION_FILE.exists():
        developed = pd.read_csv(DEVELOPED_NATION_FILE)
        developed_countries = set(developed["region"].dropna())
        countries.loc[countries["country"].isin(developed_countries), "country_group"] = (
            "developed"
        )

    return countries


def build_dynamic_rate_table(scrap_df, driver_file):
    drivers = pd.read_csv(driver_file)
    required = {
        "country_group",
        "scenario",
        "start_rate",
        "max_rate",
        "midpoint",
        "k",
        "policy_year",
        "policy_boost",
    }
    missing = required - set(drivers.columns)
    if missing:
        raise ValueError(f"{driver_file} is missing columns: {sorted(missing)}")

    countries = load_country_groups(scrap_df)
    years = sorted(scrap_df["Year"].unique())
    rows = []
    for _, country_row in countries.iterrows():
        group_drivers = drivers[drivers["country_group"] == country_row["country_group"]]
        if group_drivers.empty:
            raise ValueError(
                f"No recycling-rate drivers for group: {country_row['country_group']}"
            )
        for _, driver in group_drivers.iterrows():
            for year in years:
                rate = _logistic_value(
                    year,
                    driver["start_rate"],
                    driver["max_rate"],
                    driver["midpoint"],
                    driver["k"],
                )
                if year >= driver["policy_year"]:
                    rate += driver["policy_boost"]
                rows.append(
                    {
                        "Year": year,
                        "country": country_row["country"],
                        "country_group": country_row["country_group"],
                        "scenario": driver["scenario"],
                        "collection_rate": round(float(min(max(rate, 0), 1)), 4),
                    }
                )
    return pd.DataFrame(rows)


def build_default_rate_table(scrap_df):
    years = sorted(scrap_df["Year"].unique())
    countries = sorted(scrap_df["region"].unique())
    start_year = min(years)
    end_year = max(years)
    scenarios = {
        "baseline": (0.55, 0.75),
        "high_collection": (0.70, 0.95),
        "low_collection": (0.35, 0.60),
    }

    rows = []
    for scenario, (start_rate, end_rate) in scenarios.items():
        for year in years:
            rate = _linear_rate(year, start_year, end_year, start_rate, end_rate)
            for country in countries:
                rows.append(
                    {
                        "Year": year,
                        "country": country,
                        "scenario": scenario,
                        "collection_rate": round(rate, 4),
                    }
                )
    return pd.DataFrame(rows)


def load_rate_table(scrap_df, rate_file):
    if COUNTRY_RATE_DRIVER_FILE.exists():
        rate_df = build_dynamic_rate_table(scrap_df, COUNTRY_RATE_DRIVER_FILE)
        rate_df.to_csv(rate_file, index=False)
    elif rate_file.exists():
        rate_df = pd.read_csv(rate_file)
    else:
        rate_df = build_default_rate_table(scrap_df)
        rate_df.to_csv(rate_file, index=False)

    required = {"Year", "country", "scenario", "collection_rate"}
    missing = required - set(rate_df.columns)
    if missing:
        raise ValueError(f"{rate_file} is missing columns: {sorted(missing)}")

    rate_df = rate_df.copy()
    rate_df["collection_rate"] = pd.to_numeric(rate_df["collection_rate"], errors="raise")
    invalid = rate_df[(rate_df["collection_rate"] < 0) | (rate_df["collection_rate"] > 1)]
    if not invalid.empty:
        raise ValueError("collection_rate must be between 0 and 1")

    duplicate_keys = rate_df.duplicated(["Year", "country", "scenario"], keep=False)
    if duplicate_keys.any():
        duplicates = rate_df.loc[duplicate_keys, ["Year", "country", "scenario"]].head()
        raise ValueError(f"Duplicate recycling-rate rows found:\n{duplicates}")

    return rate_df


def load_li_content():
    metal_content = pd.read_csv(METAL_CONTENT_FILE)
    metal_content = metal_content.dropna(subset=["Type"])
    return metal_content.set_index("Type")["Li"].astype(float).to_dict()


def _parse_percent(value):
    if pd.isna(value) or value == "":
        return 0.0
    if isinstance(value, str):
        value = value.strip()
        if value.endswith("%"):
            return float(value[:-1]) / 100
    return float(value)


def load_li_recovery_efficiency():
    if TECH_RECOVERY_DRIVER_FILE.exists():
        drivers = pd.read_csv(TECH_RECOVERY_DRIVER_FILE)
        required = {
            "recycling_m",
            "recovery_efficiency_scenario",
            "start_eff",
            "max_eff",
            "midpoint",
            "k",
        }
        missing = required - set(drivers.columns)
        if missing:
            raise ValueError(
                f"{TECH_RECOVERY_DRIVER_FILE} is missing columns: {sorted(missing)}"
            )

        years = sorted(pd.read_csv(SCRAP_FILE, usecols=["Year"])["Year"].unique())
        rows = []
        for _, driver in drivers.iterrows():
            for battery_type in BATTERY_RECOVERY_ROWS:
                for year in years:
                    eff = _logistic_value(
                        year,
                        driver["start_eff"],
                        driver["max_eff"],
                        driver["midpoint"],
                        driver["k"],
                    )
                    rows.append(
                        {
                            "Year": year,
                            "type": battery_type,
                            "recycling_m": driver["recycling_m"],
                            "recovery_efficiency_scenario": driver[
                                "recovery_efficiency_scenario"
                            ],
                            "li_recovery_efficiency": round(
                                float(min(max(eff, 0), 1)), 4
                            ),
                        }
                    )
        dynamic = pd.DataFrame(rows)
        dynamic[
            [
                "Year",
                "recycling_m",
                "recovery_efficiency_scenario",
                "li_recovery_efficiency",
            ]
        ].drop_duplicates().to_csv(TECH_RECOVERY_SCENARIO_FILE, index=False)
        return dynamic

    if TECH_RECOVERY_SCENARIO_FILE.exists():
        tech = pd.read_csv(TECH_RECOVERY_SCENARIO_FILE)
        required = {
            "recycling_m",
            "recovery_efficiency_scenario",
            "li_recovery_efficiency",
        }
        missing = required - set(tech.columns)
        if missing:
            raise ValueError(
                f"{TECH_RECOVERY_SCENARIO_FILE} is missing columns: {sorted(missing)}"
            )
        tech = tech.copy()
        tech["li_recovery_efficiency"] = pd.to_numeric(
            tech["li_recovery_efficiency"], errors="raise"
        )
        invalid = tech[
            (tech["li_recovery_efficiency"] < 0)
            | (tech["li_recovery_efficiency"] > 1)
        ]
        if not invalid.empty:
            raise ValueError("li_recovery_efficiency must be between 0 and 1")

        duplicate_keys = tech.duplicated(
            ["recycling_m", "recovery_efficiency_scenario"], keep=False
        )
        if duplicate_keys.any():
            duplicates = tech.loc[
                duplicate_keys, ["recycling_m", "recovery_efficiency_scenario"]
            ].head()
            raise ValueError(f"Duplicate technology recovery rows found:\n{duplicates}")

        battery_types = pd.DataFrame({"type": list(BATTERY_RECOVERY_ROWS.keys())})
        tech["_key"] = 1
        battery_types["_key"] = 1
        return battery_types.merge(tech, on="_key").drop(columns="_key")

    recovery = pd.read_csv(RECOVERY_EFFICIENCY_FILE)
    method_cols = [col for col in recovery.columns if col != "materials"]
    rows = []

    li_product = recovery[recovery["materials"] == "Li+ in product"]
    li_product_rates = {}
    if not li_product.empty:
        li_product_rates = {
            method: _parse_percent(li_product.iloc[0][method])
            for method in method_cols
        }

    for battery_type, recovery_row in BATTERY_RECOVERY_ROWS.items():
        battery_row = recovery[recovery["materials"] == recovery_row]
        for method in method_cols:
            direct_rate = 0.0
            if not battery_row.empty:
                direct_rate = _parse_percent(battery_row.iloc[0][method])
            rows.append(
                {
                    "type": battery_type,
                    "recycling_m": method,
                    "recovery_efficiency_scenario": "existing_model",
                    "li_recovery_efficiency": max(
                        li_product_rates.get(method, 0.0), direct_rate
                    ),
                }
            )

    return pd.DataFrame(rows)


def calculate_recycling_rate_impacts(rate_file=RATE_FILE, output_dir=OUTPUT_DIR):
    scrap = pd.read_csv(SCRAP_FILE)
    rates = load_rate_table(scrap, Path(rate_file))
    li_content = load_li_content()
    li_recovery = load_li_recovery_efficiency()

    scenario_input = scrap.merge(
        rates,
        left_on=["Year", "region"],
        right_on=["Year", "country"],
        how="left",
    )
    missing_rates = scenario_input[scenario_input["collection_rate"].isna()]
    if not missing_rates.empty:
        missing = missing_rates[["Year", "region"]].drop_duplicates().head()
        raise ValueError(f"Missing collection rates for:\n{missing}")

    scenario_input["scrap_original"] = scenario_input["scrap"]
    scenario_input["scrap_collected"] = (
        scenario_input["scrap_original"] * scenario_input["collection_rate"]
    )
    scenario_input["scrap_uncollected"] = (
        scenario_input["scrap_original"] - scenario_input["scrap_collected"]
    )
    scenario_input["li_content"] = scenario_input["type"].map(li_content).fillna(0)
    scenario_input["retired_lithium"] = (
        scenario_input["scrap_original"] * scenario_input["li_content"]
    )
    scenario_input["collected_lithium"] = (
        scenario_input["scrap_collected"] * scenario_input["li_content"]
    )
    scenario_input["uncollected_lithium"] = (
        scenario_input["scrap_uncollected"] * scenario_input["li_content"]
    )

    detail = scenario_input.merge(li_recovery, on=["Year", "type"], how="left")
    detail["recycled_lithium"] = (
        detail["collected_lithium"] * detail["li_recovery_efficiency"]
    )
    detail["primary_lithium_offset"] = detail["recycled_lithium"]

    detail_cols = [
        "Year",
        "country",
        "scenario",
        "type",
        "recycling_m",
        "recovery_efficiency_scenario",
        "collection_rate",
        "scrap_original",
        "scrap_collected",
        "scrap_uncollected",
        "retired_lithium",
        "collected_lithium",
        "uncollected_lithium",
        "li_recovery_efficiency",
        "recycled_lithium",
        "primary_lithium_offset",
    ]
    detail = detail[detail_cols].sort_values(
        ["scenario", "Year", "country", "type", "recycling_m"]
    )

    country_summary = (
        detail.groupby(
            [
                "Year",
                "country",
                "scenario",
                "recovery_efficiency_scenario",
                "recycling_m",
            ],
            as_index=False,
        )
        [
            [
                "retired_lithium",
                "collected_lithium",
                "uncollected_lithium",
                "recycled_lithium",
                "primary_lithium_offset",
            ]
        ]
        .sum()
        .sort_values(
            [
                "scenario",
                "recovery_efficiency_scenario",
                "Year",
                "country",
                "recycling_m",
            ]
        )
    )
    global_summary = (
        country_summary.groupby(
            ["Year", "scenario", "recovery_efficiency_scenario", "recycling_m"],
            as_index=False,
        )
        [
            [
                "retired_lithium",
                "collected_lithium",
                "uncollected_lithium",
                "recycled_lithium",
                "primary_lithium_offset",
            ]
        ]
        .sum()
        .sort_values(["scenario", "recovery_efficiency_scenario", "Year", "recycling_m"])
    )

    collected_inputs = scenario_input.copy()
    collected_inputs["scrap"] = collected_inputs["scrap_collected"]
    collected_inputs = collected_inputs[
        [
            "Year",
            "region",
            "type",
            "scenario",
            "collection_rate",
            "scrap_original",
            "scrap",
            "scrap_uncollected",
            "inuse",
            "f_ebp",
            "tau_w",
            "f_tau",
        ]
    ].sort_values(["scenario", "Year", "region", "type"])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "lithium_recycling_rate_detail.csv"
    country_path = output_dir / "lithium_recycling_rate_country_summary.csv"
    global_path = output_dir / "lithium_recycling_rate_global_summary.csv"
    input_path = output_dir / "EV_battery_inuse_scrap_collected_by_scenario.csv"

    detail.to_csv(detail_path, index=False)
    country_summary.to_csv(country_path, index=False)
    global_summary.to_csv(global_path, index=False)
    collected_inputs.to_csv(input_path, index=False)

    return {
        "detail": detail_path,
        "country_summary": country_path,
        "global_summary": global_path,
        "collected_inputs": input_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate country recycling-rate impacts on recycled and primary lithium."
    )
    parser.add_argument(
        "--rate-file",
        default=str(RATE_FILE),
        help="CSV with Year,country,scenario,collection_rate.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory for recycling-rate scenario outputs.",
    )
    args = parser.parse_args()

    outputs = calculate_recycling_rate_impacts(args.rate_file, args.output_dir)
    for label, path in outputs.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
