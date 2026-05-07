from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SCENARIO_DIR = ROOT / "Figure_data" / "joint_policy_technology" / "lithium_loss_scenarios"
SUMMARY_FILE = SCENARIO_DIR / "lithium_loss_scenarios_summary.csv"
ROUTES_FILE = SCENARIO_DIR / "lithium_loss_scenarios_routes.csv"
COUNTRY_FILE = ROOT / "all_countries.csv"
FIGURE_DIR = ROOT / "Figure_data" / "joint_policy_technology"

YEAR = 2050
MITIGATION_SCENARIO = "baseline"
POLICIES = [
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
POLICY_COLORS = {
    "reference_policy": "#72B063",
    "current_policy": "#719AAC",
    "strict_policy": "#E29135",
    "critical_route_policy": "#CC79A7",
}
EU_COUNTRIES = {
    "Austria",
    "Belgium",
    "Bulgaria",
    "Croatia",
    "Cyprus",
    "Czech Republic",
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


def format_axes(ax, grid_axis="y"):
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9, direction="in")
    ax.grid(axis=grid_axis, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)


def clean_country(value):
    aliases = {
        "USA": "United States",
        "United States of America": "United States",
        "Korea": "Republic of Korea",
        "South Korea": "Republic of Korea",
    }
    text = str(value).strip()
    return aliases.get(text, text)


def load_country_lookup():
    countries = pd.read_csv(COUNTRY_FILE).dropna(subset=["iso3", "country"])
    countries["country_clean"] = countries["country"].map(clean_country)
    return countries.drop_duplicates("iso3").set_index("iso3")["country_clean"]


def destination_group(country):
    if country in EU_COUNTRIES:
        return "European Union"
    return country


def load_allocation():
    routes = pd.read_csv(ROUTES_FILE)
    routes = routes[
        (routes["year"] == YEAR)
        & (routes["mitigation_scenario"] == MITIGATION_SCENARIO)
        & (routes["policy_scenario"].isin(POLICIES))
        & (~routes["is_unprocessed"].astype(bool))
    ].copy()
    iso_to_country = load_country_lookup()
    routes["destination_country"] = routes["destination_iso3"].map(iso_to_country).fillna("Other")
    routes["destination_group"] = routes["destination_country"].map(destination_group)
    return (
        routes.groupby(["policy_scenario", "destination_group"], as_index=False)[
            "recovered_lithium_t"
        ]
        .sum()
        .rename(columns={"recovered_lithium_t": "recovered_li_t"})
    )


def selected_destinations(allocation, n=9):
    totals = allocation.groupby("destination_group")["recovered_li_t"].max() / 1000.0
    priority = [
        "China",
        "Republic of Korea",
        "India",
        "United States",
        "Japan",
        "European Union",
        "Canada",
        "Thailand",
    ]
    selected = [country for country in priority if country in totals.index]
    selected.extend(
        [
            country
            for country in totals.sort_values(ascending=False).index
            if country not in selected
        ][: max(0, n - len(selected))]
    )
    return selected[:n]


def allocation_table(allocation, destinations):
    table = allocation.pivot_table(
        index="policy_scenario",
        columns="destination_group",
        values="recovered_li_t",
        fill_value=0.0,
    )
    rows = []
    for policy in POLICIES:
        rows.append(
            [
                float(table.loc[policy, destination]) / 1000.0
                if policy in table.index and destination in table.columns
                else 0.0
                for destination in destinations
            ]
        )
    return pd.DataFrame(
        rows,
        index=[POLICY_LABELS[policy] for policy in POLICIES],
        columns=destinations,
    )


def plot_global_recovered(ax, summary):
    data = summary[
        (summary["year"] == YEAR)
        & (summary["mitigation_scenario"] == MITIGATION_SCENARIO)
        & (summary["policy_scenario"].isin(POLICIES))
    ].copy()
    data["policy_scenario"] = pd.Categorical(data["policy_scenario"], categories=POLICIES, ordered=True)
    data = data.sort_values("policy_scenario")
    values = data["recovered_lithium_t"].to_numpy() / 1000.0
    x = np.arange(len(data))
    bars = ax.bar(
        x,
        values,
        color=[POLICY_COLORS[p] for p in data["policy_scenario"]],
        edgecolor="black",
        linewidth=0.5,
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 7,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in data["policy_scenario"]])
    ax.set_ylabel("Global recovered Li (kt Li)", fontsize=10)
    ax.set_title("A. Policy scenarios leave global recovered Li nearly unchanged", loc="left", fontsize=11, weight="bold")
    ax.set_ylim(1600, 1685)
    format_axes(ax)


def plot_displaced(ax, summary):
    data = summary[
        (summary["year"] == YEAR)
        & (summary["mitigation_scenario"] == MITIGATION_SCENARIO)
        & (summary["policy_scenario"].isin(POLICIES))
    ].copy()
    data["policy_scenario"] = pd.Categorical(data["policy_scenario"], categories=POLICIES, ordered=True)
    data = data.sort_values("policy_scenario")
    values = data["route_access_displaced_lithium_t"].to_numpy() / 1000.0
    x = np.arange(len(data))
    bars = ax.bar(
        x,
        values,
        color=[POLICY_COLORS[p] for p in data["policy_scenario"]],
        edgecolor="black",
        linewidth=0.5,
    )
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 18,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in data["policy_scenario"]])
    ax.set_ylabel("Route-displaced recovered Li (kt Li)", fontsize=10)
    ax.set_title("B. Policy constraints displace destination access", loc="left", fontsize=11, weight="bold")
    ax.set_ylim(0, max(values) * 1.18)
    format_axes(ax)


def plot_allocation_heatmap(ax, allocation):
    destinations = selected_destinations(allocation, n=9)
    data = allocation_table(allocation, destinations)
    image = ax.imshow(data.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=30, ha="right", fontsize=8.5)
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=9)
    vmax = np.nanmax(data.values)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data.values[i, j]
            color = "white" if value > vmax * 0.55 else "black"
            label = f"{value:,.0f}" if value >= 10 else f"{value:.1f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=7.5, color=color)
    cbar = plt.colorbar(image, ax=ax, fraction=0.025, pad=0.015)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Recovered Li allocation (kt Li)", fontsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")
    ax.set_title("C. Country-scale recovered Li allocation shifts by policy", loc="left", fontsize=11, weight="bold")


def policy_delta(allocation, policy, reference="reference_policy"):
    table = allocation.pivot_table(
        index="policy_scenario",
        columns="destination_group",
        values="recovered_li_t",
        fill_value=0.0,
    )
    countries = sorted(set(table.columns))
    return pd.Series(
        {
            country: (
                float(table.loc[policy, country]) - float(table.loc[reference, country])
            )
            / 1000.0
            for country in countries
        }
    )


def plot_policy_delta(ax, allocation, policy):
    delta = policy_delta(allocation, policy)
    selected = pd.concat([delta.nsmallest(4), delta.nlargest(4)]).drop_duplicates()
    selected = selected.sort_values()
    y = np.arange(len(selected))
    colors = [POLICY_COLORS[policy] if value > 0 else "#D9D9D9" for value in selected.values]
    ax.barh(y, selected.values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=1.0)
    for yi, value in zip(y, selected.values):
        ha = "left" if value >= 0 else "right"
        offset = 10 if value >= 0 else -10
        ax.text(value + offset, yi, f"{value:+.0f}", va="center", ha=ha, fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(selected.index, fontsize=8.5)
    ax.set_xlabel("Change vs reference policy (kt Li)", fontsize=9.5)
    ax.set_title(
        f"D. {POLICY_LABELS[policy]} reallocates recovered Li across destinations",
        loc="left",
        fontsize=11,
        weight="bold",
    )
    format_axes(ax, grid_axis="x")


def main():
    plt.rcParams["font.family"] = "Arial"
    summary = pd.read_csv(SUMMARY_FILE)
    allocation = load_allocation()

    fig = plt.figure(figsize=(15.0, 9.2), dpi=320)
    grid = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.44)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    plot_global_recovered(ax_a, summary)
    plot_displaced(ax_b, summary)
    plot_allocation_heatmap(ax_c, allocation)
    plot_policy_delta(ax_d, allocation, "strict_policy")

    fig.suptitle(
        "Policy-driven country-scale allocation of recovered lithium",
        fontsize=15,
        weight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.035,
        "2050 baseline technology scenario. Policy constraints have little effect on global recovered Li, but substantially reallocate recovered Li across destination countries.",
        ha="center",
        fontsize=9.5,
        color="#374151",
    )
    fig.subplots_adjust(left=0.075, right=0.97, bottom=0.11, top=0.91)

    output_png = FIGURE_DIR / "Figure3_policy_country_allocation.png"
    output_pdf = FIGURE_DIR / "Figure3_policy_country_allocation.pdf"
    fig.savefig(output_png, bbox_inches="tight", transparent=True)
    fig.savefig(output_pdf, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Wrote {output_png}")
    print(f"Wrote {output_pdf}")


if __name__ == "__main__":
    main()
