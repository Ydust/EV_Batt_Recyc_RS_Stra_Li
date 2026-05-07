from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "Figure_data" / "joint_policy_technology"
SCENARIO_DIR = OUTPUT_DIR / "lithium_loss_scenarios"
SUMMARY_FILE = SCENARIO_DIR / "lithium_loss_scenarios_summary.csv"
ROUTES_FILE = SCENARIO_DIR / "lithium_loss_scenarios_routes.csv"
COUNTRY_FILE = ROOT / "all_countries.csv"

YEAR = 2050
POLICIES = [
    "current_policy",
    "reference_policy",
    "strict_policy",
    "critical_route_policy",
]
POLICY_LABELS = {
    "current_policy": "Current",
    "reference_policy": "Reference",
    "strict_policy": "Strict",
    "critical_route_policy": "Critical-route",
}
SCENARIOS = [
    "baseline",
    "high_direct_maturity",
    "high_recovery_efficiency",
    "combined_mitigation",
]
SCENARIO_LABELS = {
    "baseline": "Baseline",
    "high_direct_maturity": "Direct\nmaturity",
    "high_recovery_efficiency": "High\nrecovery",
    "combined_mitigation": "Combined",
}
POLICY_COLORS = {
    "current_policy": "#719AAC",
    "reference_policy": "#72B063",
    "strict_policy": "#E29135",
    "critical_route_policy": "#CC79A7",
}
TECH_COLORS = {
    "Hydro": "#72B063",
    "Direct": "#719AAC",
    "Unprocessed": "#D9D9D9",
}
COUNTRY_COLORS = {
    "China": "#E29135",
    "Republic of Korea": "#CC79A7",
    "India": "#719AAC",
    "United States": "#72B063",
    "European Union": "#E6B933",
    "Other": "#94C6CD",
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
    countries = pd.read_csv(COUNTRY_FILE)
    countries = countries.dropna(subset=["iso3", "country"]).copy()
    countries["country_clean"] = countries["country"].map(clean_country)
    return countries.drop_duplicates("iso3").set_index("iso3")["country_clean"]


def format_axes(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=10, direction="in")
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)


def plot_global_access(ax, summary):
    data = summary[
        (summary["year"] == YEAR)
        & (summary["policy_scenario"].isin(POLICIES))
        & (summary["mitigation_scenario"].isin(SCENARIOS))
    ].copy()
    table = (
        data.pivot_table(
            index="policy_scenario",
            columns="mitigation_scenario",
            values="recovered_lithium_t",
            aggfunc="mean",
        )
        .reindex(POLICIES)[SCENARIOS]
        / 1000.0
    )
    x = np.arange(len(POLICIES))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(SCENARIOS))
    colors = ["#B8DBB3", "#94C6CD", "#72B063", "#E29135"]
    for offset, scenario, color in zip(offsets, SCENARIOS, colors):
        values = table[scenario].to_numpy()
        ax.bar(
            x + offset,
            values,
            width=width,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            label=SCENARIO_LABELS[scenario].replace("\n", " "),
        )
    for xi, value in zip(x, table["combined_mitigation"].to_numpy()):
        ax.text(xi + offsets[-1], value + 18, f"{value:,.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in POLICIES], fontsize=10)
    ax.set_ylabel("Global accessible Li (kt Li)", fontsize=11)
    ax.set_title("A. Technology efficiency raises global accessible Li", loc="left", fontsize=12, weight="bold")
    ax.set_ylim(1580, 1760)
    format_axes(ax)


def plot_route_displacement(ax, summary):
    data = summary[
        (summary["year"] == YEAR)
        & (summary["policy_scenario"].isin(POLICIES))
        & (summary["mitigation_scenario"].isin(SCENARIOS))
    ].copy()
    matrix = (
        data.pivot_table(
            index="policy_scenario",
            columns="mitigation_scenario",
            values="route_access_displaced_lithium_t",
            aggfunc="mean",
        )
        .reindex(POLICIES)[SCENARIOS]
        / 1000.0
    )
    image = ax.imshow(matrix.values, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(np.arange(len(SCENARIOS)))
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS], fontsize=9)
    ax.set_yticks(np.arange(len(POLICIES)))
    ax.set_yticklabels([POLICY_LABELS[p] for p in POLICIES], fontsize=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")
    vmax = np.nanmax(matrix.values)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.values[i, j]
            color = "white" if value > vmax * 0.55 else "black"
            ax.text(j, i, f"{value:,.0f}", ha="center", va="center", fontsize=9, color=color)
    cbar = plt.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("Displaced Li on routes (kt Li)", fontsize=10)
    ax.set_title("B. Policy constraints displace otherwise accessible routes", loc="left", fontsize=12, weight="bold")


def aggregate_destination_access(routes, iso_to_country, policy, scenario):
    subset = routes[
        (routes["year"] == YEAR)
        & (routes["policy_scenario"] == policy)
        & (routes["mitigation_scenario"] == scenario)
        & (~routes["is_unprocessed"].astype(bool))
    ].copy()
    subset["country"] = subset["destination_iso3"].map(iso_to_country).fillna("Other")
    subset["country_group"] = subset["country"].where(~subset["country"].isin(EU_COUNTRIES), "European Union")
    return subset.groupby("country_group")["recovered_lithium_t"].sum() / 1000.0


def plot_destination_gain(ax, routes, iso_to_country):
    open_access = aggregate_destination_access(routes, iso_to_country, "route_access_open", "baseline")
    strict = aggregate_destination_access(routes, iso_to_country, "strict_policy", "baseline")
    delta = (strict - open_access).fillna(0.0)
    key = ["China", "Republic of Korea", "India", "United States", "European Union"]
    values = pd.Series({country: float(delta.get(country, 0.0)) for country in key})
    values["Other"] = float(delta.drop(index=[c for c in key if c in delta.index], errors="ignore").sum())
    values = values.sort_values()
    y = np.arange(len(values))
    colors = [COUNTRY_COLORS.get(country, "#9CA3AF") for country in values.index]
    ax.barh(y, values.values, color=colors, edgecolor="black", linewidth=0.5, alpha=0.82)
    ax.axvline(0, color="black", linewidth=1.0)
    for yi, value in zip(y, values.values):
        ha = "left" if value >= 0 else "right"
        offset = 8 if value >= 0 else -8
        ax.text(value + offset, yi, f"{value:+.0f}", va="center", ha=ha, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(values.index, fontsize=10)
    ax.set_xlabel("Change vs unconstrained access (kt Li)", fontsize=11)
    ax.set_title("C. Strict policy redistributes destination access", loc="left", fontsize=12, weight="bold")
    format_axes(ax)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.grid(axis="y", visible=False)


def plot_direct_share(ax, routes):
    data = routes[
        (routes["year"] == YEAR)
        & (routes["policy_scenario"].isin(POLICIES))
        & (routes["mitigation_scenario"].isin(SCENARIOS))
        & (~routes["is_unprocessed"].astype(bool))
    ].copy()
    grouped = (
        data.groupby(["policy_scenario", "mitigation_scenario", "technology"])[
            "recovered_lithium_t"
        ]
        .sum()
        .reset_index()
    )
    total = grouped.groupby(["policy_scenario", "mitigation_scenario"])[
        "recovered_lithium_t"
    ].transform("sum")
    grouped["share"] = grouped["recovered_lithium_t"] / total * 100.0
    direct = grouped[grouped["technology"] == "Direct"].pivot_table(
        index="policy_scenario",
        columns="mitigation_scenario",
        values="share",
        fill_value=0.0,
    )
    direct = direct.reindex(index=POLICIES, columns=SCENARIOS, fill_value=0.0)
    image = ax.imshow(direct.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(SCENARIOS)))
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in SCENARIOS], fontsize=9)
    ax.set_yticks(np.arange(len(POLICIES)))
    ax.set_yticklabels([POLICY_LABELS[p] for p in POLICIES], fontsize=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")
    for i in range(direct.shape[0]):
        for j in range(direct.shape[1]):
            value = direct.values[i, j]
            color = "white" if value > 58 else "black"
            ax.text(j, i, f"{value:.0f}%", ha="center", va="center", fontsize=9, color=color)
    cbar = plt.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label("Direct share of recovered Li (%)", fontsize=10)
    ax.set_title("D. Policy setting conditions Direct technology uptake", loc="left", fontsize=12, weight="bold")


def main():
    plt.rcParams["font.family"] = "Arial"
    summary = pd.read_csv(SUMMARY_FILE)
    routes = pd.read_csv(
        ROUTES_FILE,
        usecols=[
            "year",
            "policy_scenario",
            "mitigation_scenario",
            "destination_iso3",
            "technology",
            "recovered_lithium_t",
            "is_unprocessed",
        ],
    )
    iso_to_country = load_country_lookup()

    fig = plt.figure(figsize=(15.5, 9.0), dpi=320)
    grid = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.34)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    plot_global_access(ax_a, summary)
    plot_route_displacement(ax_b, summary)
    plot_destination_gain(ax_c, routes, iso_to_country)
    plot_direct_share(ax_d, routes)

    handles_a, labels_a = ax_a.get_legend_handles_labels()
    fig.legend(
        handles_a,
        labels_a,
        loc="upper center",
        bbox_to_anchor=(0.31, 0.935),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Policy constraints trade off destination access and technology-driven lithium recovery",
        fontsize=16,
        weight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.035,
        "2050 results. Technology scenarios change global recovery; policy constraints mainly redirect destination access and route flows.",
        ha="center",
        fontsize=10,
        color="#374151",
    )
    fig.subplots_adjust(left=0.07, right=0.95, bottom=0.10, top=0.88, wspace=0.30, hspace=0.42)

    output_png = OUTPUT_DIR / "Figure3_policy_technology_access_tradeoff.png"
    output_pdf = OUTPUT_DIR / "Figure3_policy_technology_access_tradeoff.pdf"
    fig.savefig(output_png, bbox_inches="tight", transparent=True)
    fig.savefig(output_pdf, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Wrote {output_png}")
    print(f"Wrote {output_pdf}")


if __name__ == "__main__":
    main()
