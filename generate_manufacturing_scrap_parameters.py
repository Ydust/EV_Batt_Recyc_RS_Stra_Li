from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PRODUCTION_CAP_FILE = ROOT / "production_cap.csv"
OUTPUT = ROOT / "manufacturing_scrap_parameters.csv"

YEARS = list(range(2010, 2051))
FORECAST_START_YEAR = 2025

COUNTRY_GROUPS = {
    "China": "China mature scale",
    "Korea": "Korea/Japan mature",
    "Japan": "Korea/Japan mature",
    "USA": "North America scaling",
    "Canada": "North America scaling",
    "Poland": "EU established hubs",
    "Hungary": "EU established hubs",
    "United Kingdom": "EU established hubs",
    "Germany": "EU ramp-up/new plants",
    "France": "EU ramp-up/new plants",
    "Sweden": "EU ramp-up/new plants",
    "Finland": "EU ramp-up/new plants",
    "Norway": "EU ramp-up/new plants",
    "Spain": "EU ramp-up/new plants",
    "Italy": "EU ramp-up/new plants",
    "India": "Emerging Asia",
    "Thailand": "Emerging Asia",
    "Viet Nam": "Emerging Asia",
    "Turkey": "Emerging Asia",
}

DEFAULT_GROUP = "Small/emerging producers"

GROUP_TARGETS = {
    "China mature scale": {
        "forecast_start_scrap": 0.07,
        "long_run_scrap": 0.04,
        "capture_2025": 0.98,
        "capture_2050": 0.99,
        "acceptance_2025": 0.985,
        "acceptance_2050": 0.995,
        "mature_historical_floor": 0.07,
    },
    "Korea/Japan mature": {
        "forecast_start_scrap": 0.06,
        "long_run_scrap": 0.04,
        "capture_2025": 0.98,
        "capture_2050": 0.99,
        "acceptance_2025": 0.985,
        "acceptance_2050": 0.995,
        "mature_historical_floor": 0.06,
    },
    "North America scaling": {
        "forecast_start_scrap": 0.12,
        "long_run_scrap": 0.05,
        "capture_2025": 0.95,
        "capture_2050": 0.98,
        "acceptance_2025": 0.965,
        "acceptance_2050": 0.990,
        "mature_historical_floor": 0.10,
    },
    "EU established hubs": {
        "forecast_start_scrap": 0.10,
        "long_run_scrap": 0.05,
        "capture_2025": 0.96,
        "capture_2050": 0.98,
        "acceptance_2025": 0.970,
        "acceptance_2050": 0.990,
        "mature_historical_floor": 0.10,
    },
    "EU ramp-up/new plants": {
        "forecast_start_scrap": 0.18,
        "long_run_scrap": 0.05,
        "capture_2025": 0.92,
        "capture_2050": 0.98,
        "acceptance_2025": 0.950,
        "acceptance_2050": 0.990,
        "mature_historical_floor": 0.12,
    },
    "Emerging Asia": {
        "forecast_start_scrap": 0.20,
        "long_run_scrap": 0.06,
        "capture_2025": 0.90,
        "capture_2050": 0.97,
        "acceptance_2025": 0.940,
        "acceptance_2050": 0.990,
        "mature_historical_floor": 0.15,
    },
    "Small/emerging producers": {
        "forecast_start_scrap": 0.22,
        "long_run_scrap": 0.07,
        "capture_2025": 0.88,
        "capture_2050": 0.96,
        "acceptance_2025": 0.930,
        "acceptance_2050": 0.988,
        "mature_historical_floor": 0.16,
    },
}

SOURCE_BASIS_HISTORICAL = (
    "Historical benchmark-calibrated estimate. Country production start and active years are from production_cap.csv. "
    "Scrap-rate levels use public battery-manufacturing yield benchmarks: Argonne BatPaC v5 assumes 95% cell yield for mature plants; "
    "Fraunhofer FFB/RWTH Aachen reports 15-30% scrap rates in early gigafactory ramp-up and around 10% even after five years; "
    "region maturity follows IEA evidence that China, Korea, and Japan dominate established cell manufacturing capacity."
)

SOURCE_BASIS_FORECAST = (
    "Forecast learning-curve estimate. Manufacturing scrap rate declines annually from the 2025 group start value toward the long-run mature value by 2050; "
    "capture and recycled-material acceptance yields improve annually with manufacturing maturity. "
    "Battery reintegration yield is calculated as (1 - manufacturing_scrap_rate) * recycled_material_acceptance_yield."
)

REFERENCE_URLS = (
    "Argonne BatPaC v5 manual: "
    "https://static.nhtsa.gov/nhtsa/downloads/CAFE/2023-NPRM-LD-2b3-2027-2035/Argonne-Databases/Documentation/BatPaC%20v5%20Manual-July%202022.pdf; "
    "Fraunhofer FFB/RWTH Aachen Mastering Ramp-up of Battery Production: "
    "https://www.ffb.fraunhofer.de/en/publications/White_papers_environment_reports_studies/Mastering_Ramp-up_of_Battery_Production.html; "
    "IEA Global EV Outlook 2024 battery recycling outlook: "
    "https://www.iea.org/reports/global-ev-outlook-2024/outlook-for-battery-and-energy-demand; "
    "IEA Recycling of Critical Minerals executive summary: "
    "https://www.iea.org/reports/recycling-of-critical-minerals/executive-summary; "
    "IEA Global EV Outlook 2025 battery supply chain concentration: "
    "https://www.iea.org/reports/global-ev-outlook-2025/electric-vehicle-batteries"
)

REFERENCE_NOTES = (
    "Argonne BatPaC provides a mature-cell-manufacturing 95% cell-yield benchmark. "
    "Fraunhofer FFB/RWTH Aachen provides early gigafactory scrap-rate range evidence of 15-30% and about 10% after five years. "
    "IEA sources support the importance of manufacturing scrap as recycling feedstock and the regional maturity distinction led by China/Korea/Japan. "
    "They do not provide direct country-year scrap rates for every country in this table."
)


def production_table():
    production = pd.read_csv(PRODUCTION_CAP_FILE)
    long = production.melt(id_vars="Year", var_name="country", value_name="production_cap")
    long["production_cap"] = pd.to_numeric(long["production_cap"], errors="coerce").fillna(0.0)
    return long


def production_countries(long):
    return sorted(long.loc[long["production_cap"] > 0, "country"].unique())


def first_active_year(long, country):
    active = long[(long["country"] == country) & (long["production_cap"] > 0)]
    if active.empty:
        return None
    return int(active["Year"].min())


def production_cap_for_year(long, country, year):
    row = long[(long["country"] == country) & (long["Year"] == year)]
    if row.empty:
        return np.nan
    return float(row.iloc[0]["production_cap"])


def historical_scrap_rate(group, year, first_year):
    if first_year is None or year < first_year:
        return np.nan
    targets = GROUP_TARGETS[group]
    age = year - first_year + 1

    if group == "China mature scale":
        if year <= 2016:
            return 0.12
        if year <= 2019:
            return 0.09
        return targets["mature_historical_floor"]
    if group == "Korea/Japan mature":
        if age <= 5:
            return 0.08
        return targets["mature_historical_floor"]

    if age <= 2:
        return 0.25
    if age <= 5:
        return 0.18
    return targets["mature_historical_floor"]


def annual_decline(year, start, target):
    if year <= FORECAST_START_YEAR:
        return start
    if year >= 2050:
        return target
    progress = (year - FORECAST_START_YEAR) / (2050 - FORECAST_START_YEAR)
    # Smooth learning curve: faster improvements early, asymptotic near target.
    curved = 1 - (1 - progress) ** 1.7
    return start + (target - start) * curved


def annual_increase(year, start, target):
    if year <= FORECAST_START_YEAR:
        return start
    if year >= 2050:
        return target
    progress = (year - FORECAST_START_YEAR) / (2050 - FORECAST_START_YEAR)
    curved = 1 - (1 - progress) ** 1.5
    return start + (target - start) * curved


def historical_capture_acceptance(group, scrap_rate):
    targets = GROUP_TARGETS[group]
    capture_rate = min(targets["capture_2025"], 0.88 + (1 - scrap_rate) * 0.10)
    acceptance_yield = min(targets["acceptance_2025"], 0.93 + (1 - scrap_rate) * 0.06)
    return capture_rate, acceptance_yield


def parameter_values(country, group, year, first_year):
    targets = GROUP_TARGETS[group]
    if year < FORECAST_START_YEAR:
        scrap_rate = historical_scrap_rate(group, year, first_year)
        if np.isnan(scrap_rate):
            capture_rate = np.nan
            acceptance_yield = np.nan
        else:
            capture_rate, acceptance_yield = historical_capture_acceptance(group, scrap_rate)
        data_status = "historical_benchmark_calibrated"
        source_basis = SOURCE_BASIS_HISTORICAL
        source_type = "benchmark_calibrated_estimate"
        is_direct_observation = False
        is_assumption = True
        assumption_method = (
            "Historical values are calibrated from country production start year, "
            "manufacturing maturity group, and public scrap/yield benchmark ranges; "
            "they are not direct country-year observed yields."
        )
    else:
        historical_2024 = historical_scrap_rate(group, 2024, first_year)
        if np.isnan(historical_2024):
            start_scrap = targets["forecast_start_scrap"]
            start_capture = targets["capture_2025"]
            start_acceptance = targets["acceptance_2025"]
        else:
            start_scrap = min(targets["forecast_start_scrap"], historical_2024)
            hist_capture, hist_acceptance = historical_capture_acceptance(group, historical_2024)
            start_capture = max(targets["capture_2025"], hist_capture)
            start_acceptance = max(targets["acceptance_2025"], hist_acceptance)
        scrap_rate = annual_decline(
            year,
            start_scrap,
            targets["long_run_scrap"],
        )
        capture_rate = annual_increase(year, start_capture, targets["capture_2050"])
        acceptance_yield = annual_increase(
            year,
            start_acceptance,
            targets["acceptance_2050"],
        )
        data_status = "forecast_learning_curve"
        source_basis = SOURCE_BASIS_FORECAST
        source_type = "forecast_assumption"
        is_direct_observation = False
        is_assumption = True
        assumption_method = (
            "Forecast values follow a smooth annual learning curve from the 2025 or 2024-calibrated level "
            "toward the long-run mature group value by 2050; they are scenario assumptions."
        )

    manufacturing_yield = np.nan if np.isnan(scrap_rate) else 1 - scrap_rate
    reintegration_yield = (
        np.nan
        if np.isnan(manufacturing_yield) or np.isnan(acceptance_yield)
        else manufacturing_yield * acceptance_yield
    )
    return {
        "manufacturing_scrap_rate": scrap_rate,
        "battery_manufacturing_yield": manufacturing_yield,
        "manufacturing_scrap_capture_rate": capture_rate,
        "recycled_material_acceptance_yield": acceptance_yield,
        "battery_reintegration_yield": reintegration_yield,
        "data_status": data_status,
        "source_type": source_type,
        "source_urls": REFERENCE_URLS,
        "reference_notes": REFERENCE_NOTES,
        "is_direct_observation": is_direct_observation,
        "is_assumption": is_assumption,
        "assumption_method": assumption_method,
        "source_basis": source_basis,
    }


def build_rows():
    production = production_table()
    rows = []
    for country in production_countries(production):
        group = COUNTRY_GROUPS.get(country, DEFAULT_GROUP)
        first_year = first_active_year(production, country)
        for year in YEARS:
            values = parameter_values(country, group, year, first_year)
            rows.append(
                {
                    "country": country,
                    "manufacturing_maturity_group": group,
                    "year": year,
                    "first_active_production_year": first_year,
                    "production_cap": production_cap_for_year(production, country, year),
                    **values,
                }
            )
    return pd.DataFrame(rows).sort_values(["country", "year"])


def main():
    data = build_rows()
    data.to_csv(OUTPUT, index=False)
    print(f"Wrote {OUTPUT}")
    print(data.groupby("manufacturing_maturity_group")["country"].nunique())
    print(data.groupby("data_status")["country"].count())


if __name__ == "__main__":
    main()
