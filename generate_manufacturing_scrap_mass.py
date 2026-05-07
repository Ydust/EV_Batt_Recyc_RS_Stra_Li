import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from generate_manufacturing_scrap_parameters import OUTPUT as PARAMETER_FILE
from generate_manufacturing_scrap_parameters import build_rows as build_parameter_rows


ROOT = Path(__file__).resolve().parent
PRODUCTION_CAP_FILE = ROOT / "production_cap.csv"
STOCK_FILE = ROOT / "Scenario result" / "EV_battery_stock.csv"
EOL_SCRAP_FILE = ROOT / "Scenario result" / "EV_battery_inuse_scrap.csv"
BNEF_WORKBOOK = ROOT / "2024-03-20 - Global Lithium-Ion Battery Supply Chain Ranking Dataset.xlsm"
BNEF_RECYCLING_OUTLOOK = ROOT / "Lithium-ion Battery Recycling Market Outlook 2024.xlsm"
OUTPUT = ROOT / "manufacturing_scrap_mass.csv"
SUMMARY_OUTPUT = ROOT / "manufacturing_scrap_mass_summary.csv"
MAIN_MODEL_OUTPUT = ROOT / "Scenario result" / "EV_battery_manufacturing_scrap.csv"
MAIN_MODEL_TOTAL_OUTPUT = ROOT / "Scenario result" / "EV_battery_manufacturing_scrap_total.csv"
COMBINED_MAIN_MODEL_OUTPUT = ROOT / "Scenario result" / "EV_battery_inuse_and_manufacturing_scrap.csv"
COMBINED_MAIN_MODEL_TOTAL_OUTPUT = ROOT / "Scenario result" / "EV_battery_inuse_and_manufacturing_scrap_total.csv"

DEFAULT_BATTERY_MASS_T_PER_KWH = 0.006
KWH_PER_GWH = 1_000_000
FORECAST_END_YEAR = 2050
BNEF_FORECAST_BASE_YEAR = 2030

BNEF_REGION_COUNTRIES = {
    "China": {"China"},
    "Japan": {"Japan"},
    "South Korea": {"Korea"},
    "US": {"USA"},
    "Europe": {
        "Belgium",
        "Czechia",
        "Finland",
        "France",
        "Germany",
        "Hungary",
        "Italy",
        "Netherlands",
        "Norway",
        "Poland",
        "Romania",
        "Russian Federation",
        "Serbia",
        "Slovakia",
        "Spain",
        "Sweden",
        "Switzerland",
        "Turkey",
        "United Kingdom",
    },
}

COUNTRY_NAME_MAP = {
    "US": "USA",
    "UK": "United Kingdom",
    "Vietnam": "Viet Nam",
    "South Korea": "Korea",
}

BNEF_CELL_COLUMNS = {
    4: 2022,
    7: 2023,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate country-year manufacturing scrap mass from cell production capacity and scrap-rate parameters."
    )
    parser.add_argument(
        "--activity-basis",
        choices=["bnef-production-scrap", "consumption-proxy", "capacity"],
        default="bnef-production-scrap",
        help=(
            "Use BNEF Outlook production scrap directly, use annual consumption/new-placement "
            "proxy as actual production volume, or use cell production capacity directly."
        ),
    )
    parser.add_argument(
        "--capacity-source",
        choices=["project", "bnef-calibrated"],
        default="bnef-calibrated",
        help=(
            "Use project production_cap.csv only, or replace 2022/2023 country values "
            "with BNEF BatteryManufacturing cell capacity where available."
        ),
    )
    parser.add_argument(
        "--capacity-forecast-method",
        choices=["hold-last"],
        default="hold-last",
        help=(
            "Method for years beyond the last positive country capacity observation. "
            "hold-last carries forward the last available capacity through the parameter horizon."
        ),
    )
    parser.add_argument(
        "--production-scrap-allocation",
        choices=["bnef-region-capacity", "global-capacity"],
        default="bnef-region-capacity",
        help=(
            "Allocate BNEF global Production scrap by BNEF region first, then by country "
            "capacity within region, or allocate directly by global country capacity share."
        ),
    )
    parser.add_argument(
        "--battery-mass-t-per-kwh",
        type=float,
        default=DEFAULT_BATTERY_MASS_T_PER_KWH,
        help="Finished battery mass intensity used to convert GWh to tonnes. Default 0.006 t/kWh, i.e. 6 kg/kWh.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT,
        help="Detailed output CSV path.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=SUMMARY_OUTPUT,
        help="Year-level summary output CSV path.",
    )
    parser.add_argument(
        "--main-model-output",
        type=Path,
        default=MAIN_MODEL_OUTPUT,
        help="Main-model-compatible manufacturing scrap CSV path.",
    )
    parser.add_argument(
        "--main-model-total-output",
        type=Path,
        default=MAIN_MODEL_TOTAL_OUTPUT,
        help="Main-model-compatible manufacturing scrap total CSV path.",
    )
    parser.add_argument(
        "--combined-main-model-output",
        type=Path,
        default=COMBINED_MAIN_MODEL_OUTPUT,
        help="Combined EOL plus manufacturing scrap CSV path using the main model schema.",
    )
    parser.add_argument(
        "--combined-main-model-total-output",
        type=Path,
        default=COMBINED_MAIN_MODEL_TOTAL_OUTPUT,
        help="Combined EOL plus manufacturing scrap total CSV path.",
    )
    return parser.parse_args()


def load_parameters():
    if PARAMETER_FILE.exists():
        params = pd.read_csv(PARAMETER_FILE)
    else:
        params = build_parameter_rows()
    params["year"] = pd.to_numeric(params["year"], errors="coerce").astype("Int64")
    return params


def load_project_capacity():
    production = pd.read_csv(PRODUCTION_CAP_FILE)
    capacity = production.melt(
        id_vars="Year",
        var_name="country",
        value_name="production_cap_gwh",
    ).rename(columns={"Year": "year"})
    capacity["year"] = pd.to_numeric(capacity["year"], errors="coerce").astype("Int64")
    capacity["production_cap_gwh"] = pd.to_numeric(
        capacity["production_cap_gwh"], errors="coerce"
    ).fillna(0.0)
    capacity["capacity_source"] = "production_cap.csv"
    return capacity


def load_bnef_capacity():
    if not BNEF_WORKBOOK.exists():
        return pd.DataFrame(columns=["country", "year", "production_cap_gwh", "capacity_source"])

    workbook = pd.ExcelFile(BNEF_WORKBOOK, engine="openpyxl")
    raw = pd.read_excel(
        workbook,
        sheet_name="BatteryManufacturing",
        header=None,
        usecols=[2, *BNEF_CELL_COLUMNS.keys()],
        skiprows=15,
        nrows=30,
    )

    rows = []
    for _, record in raw.iterrows():
        country = record.iloc[0]
        if pd.isna(country):
            continue
        country = COUNTRY_NAME_MAP.get(str(country).strip(), str(country).strip())
        for col_index, year in BNEF_CELL_COLUMNS.items():
            value = pd.to_numeric(record[col_index], errors="coerce")
            if pd.isna(value) or value <= 0:
                continue
            rows.append(
                {
                    "country": country,
                    "year": year,
                    "production_cap_gwh": float(value),
                    "capacity_source": "BNEF BatteryManufacturing cell capacity",
                }
            )
    return pd.DataFrame(rows)


def merge_capacity(capacity_source):
    capacity = load_project_capacity()
    if capacity_source == "project":
        return capacity

    bnef = load_bnef_capacity()
    if bnef.empty:
        return capacity

    key_cols = ["country", "year"]
    merged = capacity.merge(
        bnef,
        on=key_cols,
        how="left",
        suffixes=("", "_bnef"),
    )
    has_bnef = merged["production_cap_gwh_bnef"].notna()
    merged.loc[has_bnef, "production_cap_gwh"] = merged.loc[
        has_bnef, "production_cap_gwh_bnef"
    ]
    merged.loc[has_bnef, "capacity_source"] = merged.loc[has_bnef, "capacity_source_bnef"]
    return merged[["country", "year", "production_cap_gwh", "capacity_source"]]


def extend_capacity_to_parameter_horizon(capacity, parameters, method):
    if method != "hold-last":
        raise ValueError(f"Unsupported capacity forecast method: {method}")

    target_years = sorted(parameters["year"].dropna().astype(int).unique())
    countries = sorted(parameters["country"].dropna().unique())
    base = capacity.set_index(["country", "year"])
    rows = []

    for country in countries:
        country_capacity = capacity[capacity["country"] == country].copy()
        positive = country_capacity[country_capacity["production_cap_gwh"] > 0].sort_values(
            "year"
        )
        last_year = None
        last_value = 0.0
        last_source = "no_positive_capacity_observation"
        if not positive.empty:
            last = positive.iloc[-1]
            last_year = int(last["year"])
            last_value = float(last["production_cap_gwh"])
            last_source = str(last["capacity_source"])

        for year in target_years:
            key = (country, year)
            if key in base.index:
                record = base.loc[key]
                production_cap = float(record["production_cap_gwh"])
                source = str(record["capacity_source"])
            elif last_year is not None and year > last_year:
                production_cap = last_value
                source = f"forecast_hold_last_from_{last_year}; base={last_source}"
            else:
                production_cap = 0.0
                source = "no_capacity_before_first_observation"
            rows.append(
                {
                    "country": country,
                    "year": year,
                    "production_cap_gwh": production_cap,
                    "capacity_source": source,
                }
            )
    return pd.DataFrame(rows)


def load_consumption_proxy():
    stock = pd.read_csv(STOCK_FILE)
    scrap = pd.read_csv(EOL_SCRAP_FILE)
    stock["stock"] = pd.to_numeric(stock["stock"], errors="coerce").fillna(0.0)
    scrap["scrap"] = pd.to_numeric(scrap["scrap"], errors="coerce").fillna(0.0)

    keys = ["region", "type", "Year"]
    stock = stock.groupby(keys, as_index=False)["stock"].sum().sort_values(keys)
    scrap = scrap.groupby(keys, as_index=False)["scrap"].sum()
    stock["previous_stock"] = stock.groupby(["region", "type"])["stock"].shift(1).fillna(0.0)
    stock = stock.merge(scrap, on=keys, how="left")
    stock["scrap"] = stock["scrap"].fillna(0.0)
    stock["consumption_proxy_t"] = stock["stock"] - stock["previous_stock"] + stock["scrap"]
    stock["consumption_proxy_t"] = stock["consumption_proxy_t"].clip(lower=0.0)
    annual = (
        stock.groupby("Year", as_index=False)["consumption_proxy_t"]
        .sum()
        .rename(columns={"Year": "year", "consumption_proxy_t": "global_consumption_proxy_t"})
    )
    return annual


def load_type_consumption_shares():
    stock = pd.read_csv(STOCK_FILE)
    scrap = pd.read_csv(EOL_SCRAP_FILE)
    stock["stock"] = pd.to_numeric(stock["stock"], errors="coerce").fillna(0.0)
    scrap["scrap"] = pd.to_numeric(scrap["scrap"], errors="coerce").fillna(0.0)

    keys = ["region", "type", "Year"]
    stock = stock.groupby(keys, as_index=False)["stock"].sum().sort_values(keys)
    scrap = scrap.groupby(keys, as_index=False)["scrap"].sum()
    stock["previous_stock"] = stock.groupby(["region", "type"])["stock"].shift(1).fillna(0.0)
    stock = stock.merge(scrap, on=keys, how="left")
    stock["scrap"] = stock["scrap"].fillna(0.0)
    stock["type_consumption_proxy_t"] = stock["stock"] - stock["previous_stock"] + stock["scrap"]
    stock["type_consumption_proxy_t"] = stock["type_consumption_proxy_t"].clip(lower=0.0)

    annual_type = (
        stock.groupby(["Year", "type"], as_index=False)["type_consumption_proxy_t"]
        .sum()
        .rename(columns={"Year": "year"})
    )
    totals = annual_type.groupby("year")["type_consumption_proxy_t"].transform("sum")
    annual_type["type_share"] = np.where(
        totals > 0,
        annual_type["type_consumption_proxy_t"] / totals,
        0.0,
    )

    fallback = annual_type.groupby("type", as_index=False)["type_consumption_proxy_t"].sum()
    fallback_total = fallback["type_consumption_proxy_t"].sum()
    if fallback_total > 0:
        fallback["fallback_type_share"] = fallback["type_consumption_proxy_t"] / fallback_total
    else:
        fallback["fallback_type_share"] = 1.0 / len(fallback)
    annual_type = annual_type.merge(fallback[["type", "fallback_type_share"]], on="type", how="left")
    annual_type["type_share"] = np.where(
        annual_type["type_share"] > 0,
        annual_type["type_share"],
        annual_type["fallback_type_share"],
    )
    return annual_type[["year", "type", "type_share"]]


def extend_bnef_scrap_to_2050(data):
    data = data.copy()
    data["bnef_data_status"] = "BNEF_observed"
    last_year = int(data["year"].max())
    if last_year >= FORECAST_END_YEAR:
        return data

    last_value = float(data.loc[data["year"] == last_year, "bnef_production_scrap_gwh"].iloc[0])
    base = data[data["year"] == BNEF_FORECAST_BASE_YEAR]
    if base.empty:
        base_year = int(data["year"].iloc[-2])
        base_value = float(data["bnef_production_scrap_gwh"].iloc[-2])
    else:
        base_year = BNEF_FORECAST_BASE_YEAR
        base_value = float(base["bnef_production_scrap_gwh"].iloc[0])
    annual_delta = (last_value - base_value) / (last_year - base_year)

    forecast_rows = []
    for year in range(last_year + 1, FORECAST_END_YEAR + 1):
        forecast_rows.append(
            {
                "year": year,
                "bnef_production_scrap_gwh": max(0.0, last_value + annual_delta * (year - last_year)),
                "bnef_data_status": "linear_forecast_from_2030_2035_trend",
            }
        )
    return pd.concat([data, pd.DataFrame(forecast_rows)], ignore_index=True)


def load_bnef_production_scrap():
    if not BNEF_RECYCLING_OUTLOOK.exists():
        raise FileNotFoundError(BNEF_RECYCLING_OUTLOOK)

    raw = pd.read_excel(
        BNEF_RECYCLING_OUTLOOK,
        sheet_name="Figure 2",
        header=None,
        engine="openpyxl",
    )
    year_row = raw.iloc[7]
    years = [int(value) for value in year_row.iloc[2:] if pd.notna(value)]

    production_row = raw[raw.iloc[:, 1] == "Production scrap"]
    if production_row.empty:
        raise ValueError("Could not find Production scrap row in Figure 2.")
    values = production_row.iloc[0, 2 : 2 + len(years)]

    data = pd.DataFrame(
        {
            "year": years,
            "bnef_production_scrap_gwh": pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(),
        }
    )
    data = extend_bnef_scrap_to_2050(data)
    data["bnef_production_scrap_t"] = (
        data["bnef_production_scrap_gwh"] * KWH_PER_GWH * DEFAULT_BATTERY_MASS_T_PER_KWH
    )
    return data


def extend_bnef_region_shares_to_2050(region_data):
    pieces = []
    for region, group in region_data.groupby("bnef_region"):
        group = group.sort_values("year").copy()
        group["region_data_status"] = "BNEF_observed"
        last_year = int(group["year"].max())
        if last_year < FORECAST_END_YEAR:
            last_value = float(group.loc[group["year"] == last_year, "bnef_region_total_gwh"].iloc[0])
            base = group[group["year"] == BNEF_FORECAST_BASE_YEAR]
            if base.empty:
                base_year = int(group["year"].iloc[-2])
                base_value = float(group["bnef_region_total_gwh"].iloc[-2])
            else:
                base_year = BNEF_FORECAST_BASE_YEAR
                base_value = float(base["bnef_region_total_gwh"].iloc[0])
            annual_delta = (last_value - base_value) / (last_year - base_year)
            forecast_rows = []
            for year in range(last_year + 1, FORECAST_END_YEAR + 1):
                forecast_rows.append(
                    {
                        "year": year,
                        "bnef_region": region,
                        "bnef_region_total_gwh": max(
                            0.0, last_value + annual_delta * (year - last_year)
                        ),
                        "region_data_status": "linear_forecast_from_2030_2035_trend",
                    }
                )
            group = pd.concat([group, pd.DataFrame(forecast_rows)], ignore_index=True)
        pieces.append(group)
    region_data = pd.concat(pieces, ignore_index=True)
    totals = region_data.groupby("year")["bnef_region_total_gwh"].transform("sum")
    region_data["bnef_region_share"] = np.where(
        totals > 0,
        region_data["bnef_region_total_gwh"] / totals,
        0.0,
    )
    return region_data


def load_bnef_region_shares():
    raw = pd.read_excel(
        BNEF_RECYCLING_OUTLOOK,
        sheet_name="Figure 3",
        header=None,
        engine="openpyxl",
    )
    year_row = raw.iloc[8]
    years = [int(value) for value in year_row.iloc[2:] if pd.notna(value)]
    rows = []
    for region in ["China", "Europe", "Japan", "South Korea", "ROW", "US"]:
        region_row = raw[raw.iloc[:, 1] == region]
        if region_row.empty:
            raise ValueError(f"Could not find BNEF region row: {region}")
        values = pd.to_numeric(
            region_row.iloc[0, 2 : 2 + len(years)], errors="coerce"
        ).fillna(0.0)
        for year, value in zip(years, values):
            rows.append(
                {
                    "year": year,
                    "bnef_region": region,
                    "bnef_region_total_gwh": float(value),
                }
            )
    return extend_bnef_region_shares_to_2050(pd.DataFrame(rows))


def assign_bnef_region(country):
    for region, countries in BNEF_REGION_COUNTRIES.items():
        if country in countries:
            return region
    return "ROW"


def add_bnef_production_share(data, allocation_method):
    data = data.copy()
    data["bnef_region"] = data["country"].map(assign_bnef_region)
    if allocation_method == "global-capacity":
        total_capacity = data.groupby("year")["production_cap_gwh"].transform("sum")
        data["production_share"] = np.where(
            total_capacity > 0,
            data["production_cap_gwh"] / total_capacity,
            0.0,
        )
        data["bnef_region_share"] = np.nan
        data["regional_production_share"] = np.nan
        return data

    region_shares = load_bnef_region_shares()
    data = data.merge(
        region_shares[
            ["year", "bnef_region", "bnef_region_share", "region_data_status"]
        ],
        on=["year", "bnef_region"],
        how="left",
    )
    data["bnef_region_share"] = data["bnef_region_share"].fillna(0.0)
    regional_capacity = data.groupby(["year", "bnef_region"])["production_cap_gwh"].transform(
        "sum"
    )
    data["regional_production_share"] = np.where(
        regional_capacity > 0,
        data["production_cap_gwh"] / regional_capacity,
        0.0,
    )
    data["production_share"] = data["bnef_region_share"] * data["regional_production_share"]

    year_share_total = data.groupby("year")["production_share"].transform("sum")
    data["production_share"] = np.where(
        year_share_total > 0,
        data["production_share"] / year_share_total,
        0.0,
    )
    return data


def add_activity_output(data, activity_basis, battery_mass_t_per_kwh, allocation_method):
    data["activity_basis"] = activity_basis
    data["production_share"] = 0.0

    if activity_basis == "bnef-production-scrap":
        bnef_scrap = load_bnef_production_scrap()
        bnef_scrap["bnef_production_scrap_t"] = (
            bnef_scrap["bnef_production_scrap_gwh"] * KWH_PER_GWH * battery_mass_t_per_kwh
        )
        data = data.merge(bnef_scrap, on="year", how="left")
        data["bnef_production_scrap_gwh"] = data["bnef_production_scrap_gwh"].fillna(0.0)
        data["bnef_production_scrap_t"] = data["bnef_production_scrap_t"].fillna(0.0)
        data["global_consumption_proxy_t"] = np.nan
        data["global_consumption_proxy_gwh"] = np.nan
        data = add_bnef_production_share(data, allocation_method)
        data["finished_battery_output_t"] = (
            data["bnef_production_scrap_t"] * data["production_share"]
        )
        return data

    if activity_basis == "capacity":
        data["bnef_production_scrap_gwh"] = np.nan
        data["bnef_production_scrap_t"] = np.nan
        data["bnef_data_status"] = np.nan
        data["global_consumption_proxy_t"] = np.nan
        data["global_consumption_proxy_gwh"] = np.nan
        data["finished_battery_output_t"] = (
            data["production_cap_gwh"] * KWH_PER_GWH * data["battery_mass_t_per_kwh"]
        )
        return data

    consumption = load_consumption_proxy()
    data = data.merge(consumption, on="year", how="left")
    data["bnef_production_scrap_gwh"] = np.nan
    data["bnef_production_scrap_t"] = np.nan
    data["bnef_data_status"] = np.nan
    data["global_consumption_proxy_t"] = data["global_consumption_proxy_t"].fillna(0.0)
    data["global_consumption_proxy_gwh"] = (
        data["global_consumption_proxy_t"] / (KWH_PER_GWH * battery_mass_t_per_kwh)
    )
    total_capacity = data.groupby("year")["production_cap_gwh"].transform("sum")
    data["production_share"] = np.where(
        total_capacity > 0,
        data["production_cap_gwh"] / total_capacity,
        0.0,
    )
    data["finished_battery_output_t"] = (
        data["global_consumption_proxy_t"] * data["production_share"]
    )
    return data


def calculate_scrap_mass(
    parameters,
    capacity,
    battery_mass_t_per_kwh,
    activity_basis,
    allocation_method,
):
    data = parameters.drop(columns=["production_cap"], errors="ignore").merge(
        capacity,
        on=["country", "year"],
        how="left",
    )
    data["production_cap_gwh"] = pd.to_numeric(
        data["production_cap_gwh"], errors="coerce"
    ).fillna(0.0)
    data["battery_mass_t_per_kwh"] = float(battery_mass_t_per_kwh)
    data = add_activity_output(data, activity_basis, battery_mass_t_per_kwh, allocation_method)

    scrap_rate = pd.to_numeric(data["manufacturing_scrap_rate"], errors="coerce")
    yield_rate = pd.to_numeric(data["battery_manufacturing_yield"], errors="coerce")
    capture_rate = pd.to_numeric(data["manufacturing_scrap_capture_rate"], errors="coerce")
    acceptance_yield = pd.to_numeric(
        data["recycled_material_acceptance_yield"], errors="coerce"
    )

    if activity_basis == "bnef-production-scrap":
        data["gross_manufacturing_scrap_t"] = data["finished_battery_output_t"]
        data["output_basis_manufacturing_scrap_t"] = data["finished_battery_output_t"]
    else:
        valid_yield = yield_rate > 0
        data["gross_manufacturing_scrap_t"] = np.where(
            valid_yield,
            data["finished_battery_output_t"] * scrap_rate / yield_rate,
            0.0,
        )
        data["output_basis_manufacturing_scrap_t"] = (
            data["finished_battery_output_t"] * scrap_rate.fillna(0.0)
        )
    data["captured_manufacturing_scrap_t"] = (
        data["gross_manufacturing_scrap_t"] * capture_rate.fillna(0.0)
    )
    data["uncaptured_manufacturing_scrap_t"] = (
        data["gross_manufacturing_scrap_t"] - data["captured_manufacturing_scrap_t"]
    )
    data["accepted_recycled_material_t"] = (
        data["captured_manufacturing_scrap_t"] * acceptance_yield.fillna(0.0)
    )

    return data.sort_values(["country", "year"])


def summarize(data):
    return (
        data.groupby("year", as_index=False)
        .agg(
            production_cap_gwh=("production_cap_gwh", "sum"),
            global_consumption_proxy_t=("global_consumption_proxy_t", "max"),
            global_consumption_proxy_gwh=("global_consumption_proxy_gwh", "max"),
            bnef_production_scrap_gwh=("bnef_production_scrap_gwh", "max"),
            bnef_production_scrap_t=("bnef_production_scrap_t", "max"),
            bnef_data_status=("bnef_data_status", "first"),
            finished_battery_output_t=("finished_battery_output_t", "sum"),
            gross_manufacturing_scrap_t=("gross_manufacturing_scrap_t", "sum"),
            captured_manufacturing_scrap_t=("captured_manufacturing_scrap_t", "sum"),
            uncaptured_manufacturing_scrap_t=("uncaptured_manufacturing_scrap_t", "sum"),
            accepted_recycled_material_t=("accepted_recycled_material_t", "sum"),
        )
        .sort_values("year")
    )


def build_main_model_scrap(data):
    type_shares = load_type_consumption_shares()
    manufacturing = data[
        [
            "country",
            "year",
            "gross_manufacturing_scrap_t",
            "finished_battery_output_t",
            "activity_basis",
            "bnef_data_status",
        ]
    ].copy()
    manufacturing = manufacturing[manufacturing["gross_manufacturing_scrap_t"] > 0].copy()
    main = manufacturing.merge(type_shares, on="year", how="left")
    main = main.dropna(subset=["type", "type_share"])
    main["scrap"] = main["gross_manufacturing_scrap_t"] * main["type_share"]
    main["inuse"] = main["finished_battery_output_t"] * main["type_share"]
    main = main.rename(columns={"year": "Year", "country": "region"})
    main["f_ebp"] = 1.0
    main["tau_w"] = 0.0
    main["f_tau"] = 1.0
    main["source"] = "manufacturing_scrap"
    main["source_basis"] = main["activity_basis"].fillna("")
    main["data_status"] = main["bnef_data_status"].fillna("")
    return main[
        [
            "Year",
            "region",
            "type",
            "scrap",
            "inuse",
            "f_ebp",
            "tau_w",
            "f_tau",
            "source",
            "source_basis",
            "data_status",
        ]
    ].sort_values(["Year", "region", "type"])


def summarize_main_model_scrap(main):
    return (
        main.groupby(["Year", "region"], as_index=False)
        .agg(scrap=("scrap", "sum"), inuse_sum=("inuse", "sum"))
        .sort_values(["Year", "region"])
    )


def combine_with_eol(manufacturing_main):
    eol = pd.read_csv(EOL_SCRAP_FILE)
    eol["source"] = "eol_scrap"
    eol["source_basis"] = "EV_battery_inuse_scrap.csv"
    eol["data_status"] = "main_model_existing"
    combined = pd.concat([eol, manufacturing_main], ignore_index=True, sort=False)
    numeric_cols = ["scrap", "inuse", "f_ebp", "tau_w", "f_tau"]
    for col in numeric_cols:
        combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0.0)
    return combined.sort_values(["Year", "region", "type", "source"])


def main():
    args = parse_args()
    if args.battery_mass_t_per_kwh <= 0:
        raise ValueError("--battery-mass-t-per-kwh must be positive.")

    parameters = load_parameters()
    capacity = merge_capacity(args.capacity_source)
    capacity = extend_capacity_to_parameter_horizon(
        capacity, parameters, args.capacity_forecast_method
    )
    data = calculate_scrap_mass(
        parameters,
        capacity,
        args.battery_mass_t_per_kwh,
        args.activity_basis,
        args.production_scrap_allocation,
    )
    summary = summarize(data)
    main_model_scrap = build_main_model_scrap(data)
    main_model_total = summarize_main_model_scrap(main_model_scrap)
    combined_main_model = combine_with_eol(main_model_scrap)
    combined_main_model_total = summarize_main_model_scrap(combined_main_model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.main_model_output.parent.mkdir(parents=True, exist_ok=True)
    args.main_model_total_output.parent.mkdir(parents=True, exist_ok=True)
    args.combined_main_model_output.parent.mkdir(parents=True, exist_ok=True)
    args.combined_main_model_total_output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    summary.to_csv(args.summary_output, index=False)
    main_model_scrap.to_csv(args.main_model_output, index=False)
    main_model_total.to_csv(args.main_model_total_output, index=False)
    combined_main_model.to_csv(args.combined_main_model_output, index=False)
    combined_main_model_total.to_csv(args.combined_main_model_total_output, index=False)

    print(f"Wrote {args.output}")
    print(f"Wrote {args.summary_output}")
    print(f"Wrote {args.main_model_output}")
    print(f"Wrote {args.main_model_total_output}")
    print(f"Wrote {args.combined_main_model_output}")
    print(f"Wrote {args.combined_main_model_total_output}")
    print(summary[summary["year"].isin([2022, 2023, 2025, 2030])].to_string(index=False))


if __name__ == "__main__":
    main()
