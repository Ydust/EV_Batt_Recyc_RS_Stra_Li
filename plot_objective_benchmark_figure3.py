from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "Figure_data" / "joint_policy_technology" / "objective_benchmarks"
FIGURE_DIR = ROOT / "Figure_data" / "joint_policy_technology"
SUMMARY_FILE = INPUT_DIR / "objective_benchmark_summary.csv"
ROUTES_FILE = INPUT_DIR / "objective_benchmark_routes.csv"
COUNTRY_FILE = ROOT / "all_countries.csv"

RUN_ORDER = [
    ("economic_choice_baseline", "Global"),
    ("global_li_max_benchmark", "Global"),
    ("domestic_li_allocation_objective", "China"),
    ("domestic_li_allocation_objective", "United States"),
    ("domestic_li_allocation_objective", "European Union"),
]
RUN_LABELS = {
    ("economic_choice_baseline", "Global"): "Economic\nchoice",
    ("global_li_max_benchmark", "Global"): "Global\nLi-max",
    ("domestic_li_allocation_objective", "China"): "China\nallocation",
    ("domestic_li_allocation_objective", "United States"): "US\nallocation",
    ("domestic_li_allocation_objective", "European Union"): "EU\nallocation",
}
RUN_COLORS = {
    ("economic_choice_baseline", "Global"): "#B8DBB3",
    ("global_li_max_benchmark", "Global"): "#CC79A7",
    ("domestic_li_allocation_objective", "China"): "#E29135",
    ("domestic_li_allocation_objective", "United States"): "#719AAC",
    ("domestic_li_allocation_objective", "European Union"): "#E6B933",
}
DESTINATION_ORDER = [
    "China",
    "Republic of Korea",
    "India",
    "United States",
    "European Union",
    "Other",
]
DESTINATION_COLORS = {
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


def format_axes(ax, grid_axis="y"):
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9, direction="in")
    ax.grid(axis=grid_axis, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)


def run_key(row):
    return (row["objective"], row["target_region"])


def ordered_summary(summary):
    rows = []
    for objective, target_region in RUN_ORDER:
        match = summary[
            (summary["objective"] == objective)
            & (summary["target_region"] == target_region)
        ]
        if not match.empty:
            rows.append(match.iloc[0])
    return pd.DataFrame(rows)


def plot_global_recovered(ax, summary):
    data = ordered_summary(summary)
    x = np.arange(len(data))
    values = data["global_recovered_lithium_t"].to_numpy() / 1000.0
    colors = [RUN_COLORS[run_key(row)] for _, row in data.iterrows()]
    labels = [RUN_LABELS[run_key(row)] for _, row in data.iterrows()]
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
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
    ax.set_xticklabels(labels)
    ax.set_ylabel("Global recovered Li (kt Li)", fontsize=10)
    ax.set_title("A. Objective choice changes global resource efficiency", loc="left", fontsize=11, weight="bold")
    ax.set_ylim(1580, 1755)
    format_axes(ax)


def plot_target_access(ax, summary):
    data = ordered_summary(summary)
    domestic = data[data["objective"] == "domestic_li_allocation_objective"].copy()
    x = np.arange(len(domestic))
    values = domestic["target_recovered_lithium_t"].to_numpy() / 1000.0
    colors = [RUN_COLORS[run_key(row)] for _, row in domestic.iterrows()]
    labels = [row["target_region"] for _, row in domestic.iterrows()]
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 18,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Target-region recovered Li (kt Li)", fontsize=10)
    ax.set_title("B. Domestic allocation objectives prioritize regional supply", loc="left", fontsize=11, weight="bold")
    ax.set_ylim(0, max(values) * 1.16)
    format_axes(ax)


def plot_cost_penalty(ax, summary):
    data = ordered_summary(summary)
    economic_cost = float(
        data.loc[data["objective"] == "economic_choice_baseline", "route_modeled_cost"].iloc[0]
    )
    values = (data["route_modeled_cost"].to_numpy() - economic_cost) / 1e6
    x = np.arange(len(data))
    colors = [RUN_COLORS[run_key(row)] for _, row in data.iterrows()]
    labels = [RUN_LABELS[run_key(row)] for _, row in data.iterrows()]
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.2,
            f"{value:.0f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cost penalty vs economic choice (million USD)", fontsize=10)
    ax.set_title("C. Allocation and Li-max objectives impose system cost penalties", loc="left", fontsize=11, weight="bold")
    ax.set_ylim(-5, max(values) * 1.22)
    format_axes(ax)


def load_country_lookup():
    countries = pd.read_csv(COUNTRY_FILE).dropna(subset=["iso3", "country"])
    aliases = {
        "USA": "United States",
        "United States of America": "United States",
        "Korea": "Republic of Korea",
        "South Korea": "Republic of Korea",
    }
    countries["country_clean"] = countries["country"].map(
        lambda value: aliases.get(str(value).strip(), str(value).strip())
    )
    return countries.drop_duplicates("iso3").set_index("iso3")["country_clean"]


def destination_group(country):
    if country in EU_COUNTRIES:
        return "European Union"
    if country in DESTINATION_ORDER:
        return country
    return "Other"


def plot_destination_mix(ax, routes):
    iso_to_country = load_country_lookup()
    real = routes[~routes["is_unprocessed"].astype(bool)].copy()
    real["country"] = real["destination_iso3"].map(iso_to_country).fillna("Other")
    real["destination_group"] = real["country"].map(destination_group)
    grouped = (
        real.groupby(["objective", "target_region", "destination_group"])["recovered_lithium_t"]
        .sum()
        .reset_index()
    )
    total = grouped.groupby(["objective", "target_region"])["recovered_lithium_t"].transform("sum")
    grouped["share"] = grouped["recovered_lithium_t"] / total * 100.0
    table = grouped.pivot_table(
        index=["objective", "target_region"],
        columns="destination_group",
        values="share",
        fill_value=0.0,
    )
    x = np.arange(len(RUN_ORDER))
    bottom = np.zeros(len(RUN_ORDER))
    for destination in DESTINATION_ORDER:
        values = np.array(
            [
                float(table.loc[key, destination]) if key in table.index and destination in table.columns else 0.0
                for key in RUN_ORDER
            ]
        )
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=DESTINATION_COLORS[destination],
            edgecolor="black",
            linewidth=0.35,
            label=destination,
        )
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels([RUN_LABELS[key] for key in RUN_ORDER])
    ax.set_ylabel("Recovered Li allocation by destination (%)", fontsize=10)
    ax.set_title("D. Recovered Li allocation shifts across destinations", loc="left", fontsize=11, weight="bold")
    ax.set_ylim(0, 100)
    format_axes(ax)


def main():
    plt.rcParams["font.family"] = "Arial"
    summary = pd.read_csv(SUMMARY_FILE)
    routes = pd.read_csv(ROUTES_FILE)

    fig = plt.figure(figsize=(14.2, 8.4), dpi=320)
    grid = fig.add_gridspec(2, 2, wspace=0.27, hspace=0.38)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    plot_global_recovered(ax_a, summary)
    plot_target_access(ax_b, summary)
    plot_cost_penalty(ax_c, summary)
    plot_destination_mix(ax_d, routes)

    handles_d, labels_d = ax_d.get_legend_handles_labels()
    fig.legend(
        handles_d,
        labels_d,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=6,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Objective-function benchmarks for recovered lithium allocation",
        fontsize=15,
        weight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.035,
        "2050 reference-policy results under common advanced technology and capacity assumptions. Economic choice minimizes cost; Li-max maximizes global recovered Li; domestic allocation maximizes target-region recovered Li.",
        ha="center",
        fontsize=9.4,
        color="#374151",
    )
    fig.subplots_adjust(left=0.075, right=0.97, bottom=0.11, top=0.86, wspace=0.28, hspace=0.42)

    output_png = FIGURE_DIR / "Figure3_objective_benchmarks.png"
    output_pdf = FIGURE_DIR / "Figure3_objective_benchmarks.pdf"
    fig.savefig(output_png, bbox_inches="tight", transparent=True)
    fig.savefig(output_pdf, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Wrote {output_png}")
    print(f"Wrote {output_pdf}")


if __name__ == "__main__":
    main()
