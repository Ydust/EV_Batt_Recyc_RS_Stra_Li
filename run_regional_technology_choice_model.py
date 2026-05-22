from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_DIR = (
    ROOT
    / "Figure_data"
    / "joint_policy_technology"
    / "annual_dynamic_avail06_direct_x1p2"
)
OUT_DIR = ROOT / "Figure_data" / "joint_policy_technology" / "regional_technology_choice"

ROUTE_FILE = INPUT_DIR / "dynamic_scale_routes.csv"
COST_FILE = ROOT / "cost" / "cost_coun_df.csv"
COUNTRY_FILE = ROOT / "all_countries.csv"
CAPABILITY_FILE = ROOT / "technology_country_capability.csv"
DEVELOPED_FILE = ROOT / "developed_nation_list.csv"

POLICY_ORDER = [
    "reference_policy",
    "current_policy",
    "strict_policy",
    "critical_route_policy",
]
POLICY_LABELS = {
    "reference_policy": "Reference",
    "current_policy": "Current",
    "strict_policy": "Strict",
    "critical_route_policy": "Critical-route",
}
TECHNOLOGIES = ["Direct", "Hydro", "PyroHydro"]
TECH_TO_COST_COL = {"Direct": "Direct", "Hydro": "Hydro", "PyroHydro": "Pyro"}
TECH_COLORS = {"Direct": "#0072B2", "Hydro": "#009E73", "PyroHydro": "#D55E00"}
SELECTED_REGIONS = ["Global", "CHN", "KOR", "IND", "USA", "JPN", "BEL", "DEU", "CAN"]
PYROHYDRO_COST_WEIGHTS = {"Pyro": 0.45, "Hydro": 0.55}
PYROHYDRO_GROUP_COST_MULTIPLIER = {
    "developed": 1.00,
    "ev_producer": 1.06,
    "other": 1.15,
}
PYROHYDRO_COUNTRY_COST_MULTIPLIER = {
    "BEL": 0.92,
    "JPN": 0.95,
    "CAN": 0.97,
    "DEU": 1.00,
    "USA": 1.02,
    "CHN": 1.08,
    "KOR": 1.08,
    "IND": 1.18,
}

WEIGHTS = {
    "cost": 1.10,
    "capability": 1.00,
    "existing_capacity": 0.75,
    "battery_fit": 0.80,
    "scale": 0.35,
}
SOFTMAX_TEMPERATURE = 0.85

EXISTING_CAPACITY_SCORE = {
    "BEL": {"Direct": 0.25, "Hydro": 0.75, "PyroHydro": 0.95},
    "JPN": {"Direct": 0.35, "Hydro": 0.70, "PyroHydro": 0.85},
    "CAN": {"Direct": 0.25, "Hydro": 0.70, "PyroHydro": 0.80},
    "DEU": {"Direct": 0.45, "Hydro": 0.75, "PyroHydro": 0.60},
    "CHN": {"Direct": 0.85, "Hydro": 0.90, "PyroHydro": 0.25},
    "KOR": {"Direct": 0.70, "Hydro": 0.90, "PyroHydro": 0.20},
    "USA": {"Direct": 0.65, "Hydro": 0.75, "PyroHydro": 0.45},
    "IND": {"Direct": 0.30, "Hydro": 0.50, "PyroHydro": 0.15},
}
DEFAULT_EXISTING_CAPACITY = {"Direct": 0.25, "Hydro": 0.50, "PyroHydro": 0.20}

BATTERY_FIT = {
    "Direct": {
        "NMC111": 0.70,
        "NMC523": 0.80,
        "NMC622": 0.85,
        "NMC811": 0.90,
        "NCA": 0.85,
        "LFP": 0.70,
        "LMO": 0.55,
        "TLB": 0.30,
    },
    "Hydro": {
        "NMC111": 0.90,
        "NMC523": 0.92,
        "NMC622": 0.95,
        "NMC811": 0.95,
        "NCA": 0.92,
        "LFP": 0.65,
        "LMO": 0.70,
        "TLB": 0.55,
    },
    "PyroHydro": {
        "NMC111": 0.80,
        "NMC523": 0.78,
        "NMC622": 0.75,
        "NMC811": 0.72,
        "NCA": 0.75,
        "LFP": 0.35,
        "LMO": 0.70,
        "TLB": 0.85,
    },
}


def as_bool(series):
    if series.dtype == object:
        return series.astype(str).str.lower().isin(["true", "1", "yes"])
    return series.astype(bool)


def load_country_meta():
    countries = pd.read_csv(COUNTRY_FILE)
    if countries.columns[0].startswith("Unnamed") or countries.columns[0] == "H1":
        countries = countries.drop(columns=[countries.columns[0]])
    countries["producer"] = countries["producer"].astype(bool)
    countries["country_group"] = np.where(countries["producer"], "ev_producer", "other")
    if DEVELOPED_FILE.exists():
        developed = pd.read_csv(DEVELOPED_FILE)
        developed_regions = set(developed["region"].dropna())
        countries.loc[countries["country"].isin(developed_regions), "country_group"] = (
            "developed"
        )
    return countries


def interpolate_capability(year, country_group, technology, capability_table):
    method = TECH_TO_COST_COL[technology]
    group = capability_table[
        (capability_table["country_group"] == country_group)
        & (capability_table["recycling_m"] == method)
    ].sort_values("year")
    if group.empty:
        return 0.0
    availability = np.interp(year, group["year"], group["availability"])
    maturity = np.interp(year, group["year"], group["maturity_score"])
    capability = np.interp(year, group["year"], group["capability_score"])
    complexity = np.interp(year, group["year"], group["complexity_penalty"])
    policy_bonus = np.interp(year, group["year"], group["policy_bonus"])
    score = 0.35 * availability + 0.25 * maturity + 0.30 * capability + 0.10 * policy_bonus
    return float(np.clip(score - 0.20 * complexity, 0.0, 1.0))


def interpolate_scale_cost(country, method, flow_t, cost_table):
    curve = cost_table[cost_table["country"] == country].sort_values("Recycling_capacity")
    if curve.empty or method not in curve:
        return np.nan
    return float(
        np.interp(
            max(flow_t, 0.0),
            curve["Recycling_capacity"].to_numpy(dtype=float),
            curve[method].to_numpy(dtype=float),
            left=float(curve[method].iloc[0]),
            right=float(curve[method].iloc[-1]),
        )
    )


def pyrohydro_country_multiplier(iso3, country_group):
    return PYROHYDRO_COUNTRY_COST_MULTIPLIER.get(
        iso3,
        PYROHYDRO_GROUP_COST_MULTIPLIER.get(country_group, 1.10),
    )


def interpolate_unit_cost(country, technology, flow_t, cost_table, iso3=None, country_group=None):
    if technology == "PyroHydro":
        pyro_cost = interpolate_scale_cost(country, "Pyro", flow_t, cost_table)
        hydro_cost = interpolate_scale_cost(country, "Hydro", flow_t, cost_table)
        if not np.isfinite(pyro_cost) or not np.isfinite(hydro_cost):
            return np.nan
        blended_cost = (
            PYROHYDRO_COST_WEIGHTS["Pyro"] * pyro_cost
            + PYROHYDRO_COST_WEIGHTS["Hydro"] * hydro_cost
        )
        return float(blended_cost * pyrohydro_country_multiplier(iso3, country_group))
    method = TECH_TO_COST_COL[technology]
    return interpolate_scale_cost(country, method, flow_t, cost_table)


def weighted_battery_fit(group, technology):
    total = group["scrap_t"].sum()
    if total <= 0:
        return 0.0
    fit = group["battery_type"].map(BATTERY_FIT[technology]).fillna(0.55)
    return float((fit * group["scrap_t"]).sum() / total)


def softmax(values):
    values = np.asarray(values, dtype=float) / SOFTMAX_TEMPERATURE
    values = values - np.max(values)
    exp_values = np.exp(values)
    return exp_values / exp_values.sum()


def build_scores():
    routes = pd.read_csv(
        ROUTE_FILE,
        usecols=[
            "year",
            "policy_scenario",
            "destination_iso3",
            "battery_type",
            "scrap_t",
            "is_unprocessed",
        ],
    )
    routes = routes[~as_bool(routes["is_unprocessed"])].copy()
    routes["year"] = pd.to_numeric(routes["year"], errors="coerce").astype(int)
    routes["scrap_t"] = pd.to_numeric(routes["scrap_t"], errors="coerce").fillna(0.0)

    countries = load_country_meta()
    country_lookup = countries.set_index("iso3")
    cost_table = pd.read_csv(COST_FILE)
    capability_table = pd.read_csv(CAPABILITY_FILE)
    for column in [
        "year",
        "availability",
        "maturity_score",
        "capability_score",
        "complexity_penalty",
        "policy_bonus",
    ]:
        capability_table[column] = pd.to_numeric(capability_table[column], errors="coerce")

    flow = (
        routes.groupby(["year", "policy_scenario", "destination_iso3"], as_index=False)[
            "scrap_t"
        ]
        .sum()
        .rename(columns={"destination_iso3": "region", "scrap_t": "flow_t"})
    )
    battery_groups = {
        key: group
        for key, group in routes.groupby(["year", "policy_scenario", "destination_iso3"])
    }
    max_flow_by_year = flow.groupby("year")["flow_t"].max().to_dict()

    rows = []
    for _, flow_row in flow.iterrows():
        year = int(flow_row["year"])
        policy = flow_row["policy_scenario"]
        iso3 = flow_row["region"]
        if iso3 not in country_lookup.index:
            continue
        country_row = country_lookup.loc[iso3]
        country = country_row["country"]
        group_name = country_row["country_group"]
        flow_t = float(flow_row["flow_t"])
        battery_group = battery_groups[(year, policy, iso3)]

        raw_costs = {
            tech: interpolate_unit_cost(
                country,
                tech,
                flow_t,
                cost_table,
                iso3=iso3,
                country_group=group_name,
            )
            for tech in TECHNOLOGIES
        }
        valid_costs = [value for value in raw_costs.values() if np.isfinite(value)]
        cost_min = min(valid_costs) if valid_costs else 0.0
        cost_max = max(valid_costs) if valid_costs else 1.0
        cost_range = max(cost_max - cost_min, 1e-9)
        utilities = []
        tech_rows = []
        for tech in TECHNOLOGIES:
            cost = raw_costs[tech]
            normalized_cost = (cost - cost_min) / cost_range if np.isfinite(cost) else 1.0
            cost_score = 1.0 - normalized_cost
            capability = interpolate_capability(year, group_name, tech, capability_table)
            existing_capacity = EXISTING_CAPACITY_SCORE.get(
                iso3, DEFAULT_EXISTING_CAPACITY
            ).get(tech, DEFAULT_EXISTING_CAPACITY[tech])
            battery_fit = weighted_battery_fit(battery_group, tech)
            scale_score = np.log1p(flow_t) / np.log1p(max_flow_by_year.get(year, flow_t))
            utility = (
                WEIGHTS["cost"] * cost_score
                + WEIGHTS["capability"] * capability
                + WEIGHTS["existing_capacity"] * existing_capacity
                + WEIGHTS["battery_fit"] * battery_fit
                + WEIGHTS["scale"] * scale_score
            )
            utilities.append(utility)
            tech_rows.append(
                {
                    "year": year,
                    "policy_scenario": policy,
                    "region": iso3,
                    "country": country,
                    "technology": tech,
                    "flow_t": flow_t,
                    "unit_cost": cost,
                    "cost_score": cost_score,
                    "capability_score": capability,
                    "existing_capacity_score": existing_capacity,
                    "battery_fit_score": battery_fit,
                    "scale_score": scale_score,
                    "utility": utility,
                }
            )
        shares = softmax(utilities)
        for tech_row, share in zip(tech_rows, shares):
            tech_row["technology_share_pct"] = float(share * 100.0)
            tech_row["technology_throughput_t"] = float(flow_t * share)
            rows.append(tech_row)

    return pd.DataFrame(rows)


def add_global_rows(scores):
    region_rows = [scores]
    global_rows = (
        scores.groupby(["year", "policy_scenario", "technology"], as_index=False)[
            "technology_throughput_t"
        ]
        .sum()
    )
    totals = (
        scores.drop_duplicates(["year", "policy_scenario", "region"])
        .groupby(["year", "policy_scenario"], as_index=False)["flow_t"]
        .sum()
        .rename(columns={"flow_t": "global_flow_t"})
    )
    global_rows = global_rows.merge(totals, on=["year", "policy_scenario"], how="left")
    global_rows["technology_share_pct"] = (
        global_rows["technology_throughput_t"] / global_rows["global_flow_t"] * 100.0
    )
    global_rows["region"] = "Global"
    global_rows["country"] = "Global"
    global_rows["flow_t"] = global_rows["global_flow_t"]
    for column in [
        "unit_cost",
        "cost_score",
        "capability_score",
        "existing_capacity_score",
        "battery_fit_score",
        "scale_score",
        "utility",
    ]:
        global_rows[column] = np.nan
    region_rows.append(global_rows.drop(columns=["global_flow_t"]))
    return pd.concat(region_rows, ignore_index=True)


def plot_global_trend(shares):
    plt.rcParams.update({"font.family": "Arial"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=300, sharex=True, sharey=True)
    global_data = shares[shares["region"] == "Global"].copy()
    for ax, policy in zip(axes.flat, POLICY_ORDER):
        subset = global_data[global_data["policy_scenario"] == policy]
        for tech in TECHNOLOGIES:
            line = subset[subset["technology"] == tech].sort_values("year")
            ax.plot(
                line["year"],
                line["technology_share_pct"],
                label=tech,
                color=TECH_COLORS[tech],
                linewidth=2.2,
            )
        ax.set_title(POLICY_LABELS[policy], fontsize=13, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.set_xlim(2030, 2050)
        ax.set_xticks([2030, 2035, 2040, 2045, 2050])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color="0.88")
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.0)
    axes[0, 0].set_ylabel("Technology share (%)")
    axes[1, 0].set_ylabel("Technology share (%)")
    axes[1, 0].set_xlabel("Year")
    axes[1, 1].set_xlabel("Year")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        prop={"size": 12},
    )
    fig.suptitle("Global technology shares from regional choice model", fontsize=15)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(OUT_DIR / "regional_technology_choice_global_trend.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "regional_technology_choice_global_trend.pdf", bbox_inches="tight")


def plot_selected_regions(shares):
    plt.rcParams.update({"font.family": "Arial"})
    selected = shares[shares["region"].isin(SELECTED_REGIONS)].copy()
    selected["region_label"] = selected["region"].replace({"Global": "Global"})
    fig, axes = plt.subplots(3, 3, figsize=(15, 10), dpi=300, sharex=True, sharey=True)
    for ax, region in zip(axes.flat, SELECTED_REGIONS):
        subset = selected[
            (selected["region"] == region)
            & (selected["policy_scenario"] == "reference_policy")
        ]
        for tech in TECHNOLOGIES:
            line = subset[subset["technology"] == tech].sort_values("year")
            ax.plot(
                line["year"],
                line["technology_share_pct"],
                color=TECH_COLORS[tech],
                linewidth=2.0,
                label=tech,
            )
        country_name = (
            "Global"
            if region == "Global"
            else subset["country"].dropna().iloc[0]
            if not subset.empty
            else region
        )
        ax.set_title(country_name, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 100)
        ax.set_xlim(2030, 2050)
        ax.set_xticks([2030, 2035, 2040, 2045, 2050])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color="0.9")
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.0)
    for ax in axes[:, 0]:
        ax.set_ylabel("Share (%)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Year")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        prop={"size": 12},
    )
    fig.suptitle(
        "Selected region technology shares under reference policy",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(
        OUT_DIR / "regional_technology_choice_selected_regions_reference.png",
        bbox_inches="tight",
    )
    fig.savefig(
        OUT_DIR / "regional_technology_choice_selected_regions_reference.pdf",
        bbox_inches="tight",
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = build_scores()
    shares = add_global_rows(scores)
    scores.to_csv(OUT_DIR / "regional_technology_choice_scores.csv", index=False)
    shares.to_csv(OUT_DIR / "regional_technology_choice_shares.csv", index=False)
    plot_global_trend(shares)
    plot_selected_regions(shares)
    print(f"Wrote {OUT_DIR / 'regional_technology_choice_scores.csv'}")
    print(f"Wrote {OUT_DIR / 'regional_technology_choice_shares.csv'}")
    print(f"Wrote {OUT_DIR / 'regional_technology_choice_global_trend.png'}")
    print(f"Wrote {OUT_DIR / 'regional_technology_choice_selected_regions_reference.png'}")


if __name__ == "__main__":
    main()
