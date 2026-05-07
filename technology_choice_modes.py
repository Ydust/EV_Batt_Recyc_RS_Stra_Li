import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
METAL_CONTENT_FILE = ROOT / "cost" / "Metal content.csv"
RECYCLING_EMISSION_FILE = ROOT / "cost" / "recycling_emission_count.csv"
TECH_RECOVERY_FILE = ROOT / "technology_lithium_recovery_scenarios.csv"
ALL_COUNTRIES_FILE = ROOT / "all_countries.csv"
DEVELOPED_NATION_FILE = ROOT / "developed_nation_list.csv"
TECH_CAPABILITY_FILE = ROOT / "technology_country_capability.csv"
BATTERY_FIT_FILE = ROOT / "technology_battery_fit.csv"
LITHIUM_PRICE_FILE = ROOT / "lithium_price_scenario.csv"


def normalize(series):
    series = pd.to_numeric(series, errors="coerce").fillna(0)
    span = series.max() - series.min()
    if span == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.min()) / span


def load_support_tables(recovery_scenario):
    metal = pd.read_csv(METAL_CONTENT_FILE)
    li_content = metal.dropna(subset=["Type"]).set_index("Type")["Li"].astype(float)

    recovery = pd.read_csv(TECH_RECOVERY_FILE)
    recovery = recovery[
        recovery["recovery_efficiency_scenario"] == recovery_scenario
    ].copy()
    if recovery.empty:
        raise ValueError(f"No recovery-efficiency scenario found: {recovery_scenario}")

    emission = pd.read_csv(RECYCLING_EMISSION_FILE)
    emission = emission.rename(columns={"battery_type": "type"})
    emission["recycling_CO2_per_t"] = pd.to_numeric(emission["CO2"], errors="coerce") / 1000
    emission = emission[["type", "recycling_m", "recycling_CO2_per_t"]]
    country_groups = load_country_groups()
    capability = load_capability_table()
    battery_fit = pd.read_csv(BATTERY_FIT_FILE)
    lithium_price = load_lithium_price_table()
    return li_content, recovery, emission, country_groups, capability, battery_fit, lithium_price


def load_lithium_price_table(price_scenario="baseline"):
    price = pd.read_csv(LITHIUM_PRICE_FILE)
    price = price[price["price_scenario"] == price_scenario].copy()
    if price.empty:
        raise ValueError(f"No lithium price scenario found: {price_scenario}")
    price["Year"] = pd.to_numeric(price["Year"], errors="raise")
    price["lithium_price_usd_per_t"] = pd.to_numeric(
        price["lithium_price_usd_per_t"], errors="raise"
    )
    return price[["Year", "lithium_price_usd_per_t"]]


def load_country_groups():
    all_countries = pd.read_csv(ALL_COUNTRIES_FILE)
    country_groups = all_countries[["country", "producer"]].copy()
    country_groups["country_group"] = np.where(
        country_groups["producer"], "ev_producer", "other"
    )
    if DEVELOPED_NATION_FILE.exists():
        developed = pd.read_csv(DEVELOPED_NATION_FILE)
        developed_countries = set(developed["region"].dropna())
        country_groups.loc[
            country_groups["country"].isin(developed_countries), "country_group"
        ] = "developed"
    return country_groups[["country", "country_group"]]


def load_capability_table():
    capability = pd.read_csv(TECH_CAPABILITY_FILE)
    numeric_cols = [
        "year",
        "availability",
        "maturity_score",
        "capability_score",
        "complexity_penalty",
        "policy_bonus",
    ]
    for col in numeric_cols:
        capability[col] = pd.to_numeric(capability[col], errors="raise")
    return capability


def interpolate_capability(capability, years):
    rows = []
    value_cols = [
        "availability",
        "maturity_score",
        "capability_score",
        "complexity_penalty",
        "policy_bonus",
    ]
    for (country_group, recycling_m), group in capability.groupby(
        ["country_group", "recycling_m"]
    ):
        group = group.sort_values("year")
        for year in years:
            row = {
                "country_group": country_group,
                "recycling_m": recycling_m,
                "year": year,
            }
            for col in value_cols:
                row[col] = np.interp(year, group["year"], group[col])
            rows.append(row)
    return pd.DataFrame(rows)


def read_strategy_candidates(result_root, year, strategy):
    path = result_root / str(year) / f"net_profit_all_{strategy}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path, index_col=0)
    data["year"] = year
    data["Strategy type"] = strategy
    return data


def add_lithium_and_emission_metrics(
    data,
    li_content,
    recovery,
    emission,
    country_groups,
    capability,
    battery_fit,
    lithium_price,
):
    data = data.copy()
    data["li_content"] = data["type"].map(li_content).fillna(0)
    data["contained_lithium"] = data["scrap"] * data["li_content"]
    data = data.merge(
        recovery[["Year", "recycling_m", "li_recovery_efficiency"]],
        left_on=["year", "recycling_m"],
        right_on=["Year", "recycling_m"],
        how="left",
    ).drop(columns=["Year"])
    data["li_recovery_efficiency"] = data["li_recovery_efficiency"].fillna(0)
    data["recycled_lithium"] = (
        data["contained_lithium"] * data["li_recovery_efficiency"]
    )
    data["primary_lithium_offset"] = data["recycled_lithium"]
    data["primary_lithium_gap"] = data["contained_lithium"] - data["recycled_lithium"]

    data = data.merge(emission, on=["type", "recycling_m"], how="left")
    data["recycling_CO2_per_t"] = data["recycling_CO2_per_t"].fillna(0)
    data["recycling_CO2_em"] = data["recycling_CO2_per_t"] * data["scrap"]
    data["total_netprofits"] = data["scrap"] * data["net_profit"] * 1000
    data["total_costs"] = data["scrap"] * data["unit_cost"] * 1000
    data = data.merge(
        lithium_price,
        left_on="year",
        right_on="Year",
        how="left",
    ).drop(columns=["Year"])
    data["lithium_price_usd_per_t"] = data["lithium_price_usd_per_t"].interpolate().ffill().bfill()
    data["dynamic_lithium_revenue"] = (
        data["recycled_lithium"] * data["lithium_price_usd_per_t"]
    )
    data["dynamic_total_netprofits"] = (
        data["total_netprofits"] + data["dynamic_lithium_revenue"]
    )
    data["dynamic_net_profit"] = np.where(
        data["scrap"] > 0,
        data["dynamic_total_netprofits"] / data["scrap"] / 1000,
        0,
    )

    data = data.merge(country_groups, on="country", how="left")
    data["country_group"] = data["country_group"].fillna("other")
    capability_years = interpolate_capability(capability, sorted(data["year"].unique()))
    data = data.merge(
        capability_years,
        on=["country_group", "recycling_m", "year"],
        how="left",
    )
    data = data.merge(
        battery_fit[["type", "recycling_m", "battery_type_fit"]],
        on=["type", "recycling_m"],
        how="left",
    )
    for col in [
        "availability",
        "maturity_score",
        "capability_score",
        "complexity_penalty",
        "policy_bonus",
        "battery_type_fit",
    ]:
        data[col] = data[col].fillna(0)
    return data


def choose_by_mode(candidates, alpha, beta, availability_threshold):
    rows = []
    group_cols = ["year", "Strategy type", "country", "type"]
    for _, group in candidates.groupby(group_cols, dropna=False):
        group = group.copy()
        if group["scrap"].sum() <= 0:
            group["score_profit"] = 0
            group["score_lithium"] = 0
            group["score_multiobjective"] = 0
            group["score_realistic_multiobjective"] = 0
        else:
            group["score_profit"] = group["net_profit"]
            group["score_lithium"] = group["recycled_lithium"]
            group["score_multiobjective"] = (
                normalize(group["dynamic_net_profit"])
                + alpha * normalize(group["recycled_lithium"])
                - beta * normalize(group["recycling_CO2_em"])
            )
            group["score_realistic_multiobjective"] = (
                normalize(group["dynamic_net_profit"])
                + alpha * normalize(group["recycled_lithium"])
                - beta * normalize(group["recycling_CO2_em"])
                + 0.65 * group["maturity_score"]
                + 0.65 * group["capability_score"]
                + 0.45 * group["battery_type_fit"]
                + 0.35 * group["policy_bonus"]
                - 0.55 * group["complexity_penalty"]
            )

        mode_to_score = {
            "Optimal_profit": "score_profit",
            "Optimal_lithium": "score_lithium",
            "Optimal_multiobjective": "score_multiobjective",
            "Realistic_multiobjective": "score_realistic_multiobjective",
        }
        for mode, score_col in mode_to_score.items():
            selectable = group.copy()
            if mode == "Realistic_multiobjective":
                selectable = selectable[
                    selectable["availability"] >= availability_threshold
                ].copy()
                if selectable.empty:
                    selectable = group.copy()
            choice = selectable.loc[selectable[score_col].idxmax()].copy()
            choice["choice_mode"] = mode
            choice["choice_score"] = choice[score_col]
            rows.append(choice)
    return pd.DataFrame(rows)


def summarize_choices(choices):
    country_summary = (
        choices.groupby(
            [
                "year",
                "Strategy type",
                "choice_mode",
                "country",
                "recycling_m",
            ],
            as_index=False,
        )[
            [
                "scrap",
                "contained_lithium",
                "recycled_lithium",
                "primary_lithium_offset",
                "primary_lithium_gap",
                "recycling_CO2_em",
                "total_netprofits",
                "total_costs",
            ]
        ]
        .sum()
        .sort_values(["year", "Strategy type", "choice_mode", "country", "recycling_m"])
    )
    global_summary = (
        country_summary.groupby(
            ["year", "Strategy type", "choice_mode", "recycling_m"], as_index=False
        )[
            [
                "scrap",
                "contained_lithium",
                "recycled_lithium",
                "primary_lithium_offset",
                "primary_lithium_gap",
                "recycling_CO2_em",
                "total_netprofits",
                "total_costs",
            ]
        ]
        .sum()
        .sort_values(["year", "Strategy type", "choice_mode", "recycling_m"])
    )
    return country_summary, global_summary


def run_choice_modes(
    collection_scenario,
    recovery_scenario,
    year_start,
    year_end,
    years,
    alpha,
    beta,
    availability_threshold,
):
    result_root = (
        ROOT
        / "trans"
        / "scenario_result"
        / collection_scenario
        / recovery_scenario
    )
    output_root = result_root / "technology_choice_modes"
    output_root.mkdir(parents=True, exist_ok=True)

    li_content, recovery, emission, country_groups, capability, battery_fit, lithium_price = (
        load_support_tables(recovery_scenario)
    )
    all_candidates = []
    year_list = years or list(range(year_start, year_end + 1))
    for year in year_list:
        for strategy in ["Strategy 1", "Strategy 2", "Strategy 3"]:
            candidates = read_strategy_candidates(result_root, year, strategy)
            all_candidates.append(candidates)

    candidates = pd.concat(all_candidates, ignore_index=True)
    candidates = add_lithium_and_emission_metrics(
        candidates,
        li_content,
        recovery,
        emission,
        country_groups,
        capability,
        battery_fit,
        lithium_price,
    )
    choices = choose_by_mode(candidates, alpha, beta, availability_threshold)
    country_summary, global_summary = summarize_choices(choices)

    candidates.to_csv(output_root / "technology_choice_candidates.csv", index=False)
    choices.to_csv(output_root / "technology_choice_by_mode.csv", index=False)
    country_summary.to_csv(
        output_root / "technology_choice_country_summary.csv", index=False
    )
    global_summary.to_csv(
        output_root / "technology_choice_global_summary.csv", index=False
    )
    return output_root


def main():
    parser = argparse.ArgumentParser(
        description="Post-process Strategy 1/2/3 outputs into profit, lithium, and multi-objective technology-choice modes."
    )
    parser.add_argument("--collection-scenario", default="high_collection")
    parser.add_argument("--recovery-scenario", default="baseline")
    parser.add_argument("--year-start", type=int, default=2025)
    parser.add_argument("--year-end", type=int, default=2025)
    parser.add_argument(
        "--years",
        default="",
        help="Comma-separated non-contiguous years, e.g. 2025,2030,2035.",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--availability-threshold", type=float, default=0.5)
    args = parser.parse_args()

    output_root = run_choice_modes(
        args.collection_scenario,
        args.recovery_scenario,
        args.year_start,
        args.year_end,
        [int(y.strip()) for y in args.years.split(",") if y.strip()] if args.years else None,
        args.alpha,
        args.beta,
        args.availability_threshold,
    )
    print(output_root)


if __name__ == "__main__":
    main()
