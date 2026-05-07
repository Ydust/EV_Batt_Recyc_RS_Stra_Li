from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "Figure_data" / "joint_policy_technology" / "objective_benchmarks"
FIGURE_DIR = ROOT / "Figure_data" / "joint_policy_technology"
ROUTES_FILE = INPUT_DIR / "objective_benchmark_routes.csv"
SUMMARY_FILE = INPUT_DIR / "objective_benchmark_summary.csv"
COUNTRY_FILE = ROOT / "all_countries.csv"

OBJECTIVE_ORDER = [
    ("economic_choice_baseline", "Global"),
    ("global_li_max_benchmark", "Global"),
    ("domestic_li_allocation_objective", "China"),
    ("domestic_li_allocation_objective", "United States"),
    ("domestic_li_allocation_objective", "European Union"),
]
OBJECTIVE_LABELS = {
    ("economic_choice_baseline", "Global"): "Economic\nchoice",
    ("global_li_max_benchmark", "Global"): "Global\nLi-max",
    ("domestic_li_allocation_objective", "China"): "China\nallocation",
    ("domestic_li_allocation_objective", "United States"): "US\nallocation",
    ("domestic_li_allocation_objective", "European Union"): "EU\nallocation",
}
TARGET_COLORS = {
    "China": "#E29135",
    "United States": "#719AAC",
    "European Union": "#E6B933",
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


def format_axes(ax, grid_axis="x"):
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


def country_group(country):
    if country in EU_COUNTRIES:
        return "European Union"
    return country


def load_allocation():
    routes = pd.read_csv(ROUTES_FILE)
    routes = routes[~routes["is_unprocessed"].astype(bool)].copy()
    iso_to_country = load_country_lookup()
    routes["destination_country"] = routes["destination_iso3"].map(iso_to_country).fillna("Other")
    routes["destination_group"] = routes["destination_country"].map(country_group)
    grouped = (
        routes.groupby(["objective", "target_region", "destination_group"], as_index=False)[
            "recovered_lithium_t"
        ]
        .sum()
        .rename(columns={"recovered_lithium_t": "recovered_li_t"})
    )
    return grouped


def top_destinations(allocation, n=10):
    total = allocation.groupby("destination_group")["recovered_li_t"].max()
    priority = ["China", "United States", "European Union", "Republic of Korea", "India"]
    selected = [country for country in priority if country in total.index]
    remaining = total.drop(index=selected, errors="ignore").sort_values(ascending=False)
    selected.extend([country for country in remaining.index if country not in selected][: max(0, n - len(selected))])
    return selected[:n]


def pivot_allocation_kt(allocation, destinations):
    table = allocation.pivot_table(
        index=["objective", "target_region"],
        columns="destination_group",
        values="recovered_li_t",
        fill_value=0.0,
    )
    values = []
    for key in OBJECTIVE_ORDER:
        row = []
        for destination in destinations:
            row.append(float(table.loc[key, destination]) / 1000.0 if key in table.index and destination in table.columns else 0.0)
        values.append(row)
    return pd.DataFrame(values, index=[OBJECTIVE_LABELS[key].replace("\n", " ") for key in OBJECTIVE_ORDER], columns=destinations)


def plot_allocation_heatmap(ax, allocation):
    destinations = top_destinations(allocation, n=9)
    data = pivot_allocation_kt(allocation, destinations)
    image = ax.imshow(data.values, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=35, ha="right", fontsize=8.5)
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=9)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data.values[i, j]
            color = "white" if value > np.nanmax(data.values) * 0.55 else "black"
            label = f"{value:,.0f}" if value >= 10 else f"{value:.1f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=7.5, color=color)
    cbar = plt.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Recovered Li allocation (kt Li)", fontsize=9)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")
    ax.set_title("A. National allocation patterns reveal where recovered Li goes", loc="left", fontsize=11, weight="bold")


def objective_delta(allocation, objective, target_region, reference_objective, reference_region):
    baseline = allocation[
        (allocation["objective"] == reference_objective)
        & (allocation["target_region"] == reference_region)
    ].set_index("destination_group")["recovered_li_t"]
    target = allocation[
        (allocation["objective"] == objective)
        & (allocation["target_region"] == target_region)
    ].set_index("destination_group")["recovered_li_t"]
    countries = sorted(set(baseline.index) | set(target.index))
    delta = pd.Series({country: float(target.get(country, 0.0) - baseline.get(country, 0.0)) / 1000.0 for country in countries})
    return delta


def plot_allocation_shifts(
    ax,
    allocation,
    objective,
    target_region,
    reference_objective,
    reference_region,
    title,
    color,
    min_abs_value=0.5,
):
    delta = objective_delta(
        allocation,
        objective,
        target_region,
        reference_objective,
        reference_region,
    )
    delta = delta[delta.abs() >= min_abs_value]
    if delta.empty:
        ax.text(0.5, 0.5, "No material shift", transform=ax.transAxes, ha="center", va="center", fontsize=11)
        ax.set_title(title, loc="left", fontsize=10.5, weight="bold")
        ax.set_axis_off()
        return
    selected = pd.concat([delta.nsmallest(4), delta.nlargest(4)]).drop_duplicates()
    selected = selected.sort_values()
    y = np.arange(len(selected))
    colors = [color if value > 0 else "#D9D9D9" for value in selected.values]
    ax.barh(y, selected.values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=1.0)
    span = max(abs(float(selected.min())), abs(float(selected.max())), 1.0)
    ax.set_xlim(-span * 1.18, span * 1.18)
    offset = span * 0.025
    for yi, value in zip(y, selected.values):
        ha = "left" if value >= 0 else "right"
        label_offset = offset if value >= 0 else -offset
        ax.text(
            value + label_offset,
            yi,
            f"{value:+.0f}",
            va="center",
            ha=ha,
            fontsize=8.5,
            clip_on=True,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(selected.index, fontsize=8.5)
    ax.set_xlabel("Recovered Li allocation shift (kt Li)", fontsize=9.5)
    ax.set_title(title, loc="left", fontsize=10.5, weight="bold")
    format_axes(ax, grid_axis="x")


def plot_tradeoff(ax, summary):
    global_max = summary[
        (summary["objective"] == "global_li_max_benchmark")
        & (summary["target_region"] == "Global")
    ].iloc[0]
    allocation = load_allocation()
    global_allocation = allocation[
        (allocation["objective"] == "global_li_max_benchmark")
        & (allocation["target_region"] == "Global")
    ].set_index("destination_group")["recovered_li_t"]
    rows = []
    for _, row in summary[summary["objective"] == "domestic_li_allocation_objective"].iterrows():
        target_region = row["target_region"]
        baseline_target_li = float(global_allocation.get(target_region, 0.0))
        rows.append(
            {
                "target_region": target_region,
                "target_allocation_increase_kt": (
                    float(row["target_recovered_lithium_t"]) - baseline_target_li
                )
                / 1000.0,
                "cost_penalty_m": (
                    float(row["route_modeled_cost"])
                    - float(global_max["route_modeled_cost"])
                )
                / 1e6,
            }
        )
    data = pd.DataFrame(rows)
    for _, row in data.iterrows():
        color = TARGET_COLORS.get(row["target_region"], "#94C6CD")
        ax.scatter(
            row["target_allocation_increase_kt"],
            row["cost_penalty_m"],
            s=120,
            color=color,
            edgecolor="black",
            linewidth=0.8,
        )
        label_offsets = {
            "China": (14, 1.2),
            "United States": (14, -1.4),
            "European Union": (12, 1.0),
        }
        dx, dy = label_offsets.get(row["target_region"], (12, 1.0))
        ax.text(
            row["target_allocation_increase_kt"] + dx,
            row["cost_penalty_m"] + dy,
            row["target_region"],
            fontsize=9,
        )
    ax.axhline(0, color="black", linewidth=1.0)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_xlabel("Target-region allocation increase vs Global Li-max (kt Li)", fontsize=9.5)
    ax.set_ylabel("Additional cost vs Global Li-max (million USD)", fontsize=9.5)
    ax.set_title("E. Additional regional allocation relative to Global Li-max", loc="left", fontsize=10.5, weight="bold")
    ax.set_xlim(-20, data["target_allocation_increase_kt"].max() * 1.22)
    ax.set_ylim(min(-3, data["cost_penalty_m"].min() * 1.2), data["cost_penalty_m"].max() * 1.25)
    format_axes(ax, grid_axis="both")


def main():
    plt.rcParams["font.family"] = "Arial"
    allocation = load_allocation()
    summary = pd.read_csv(SUMMARY_FILE)

    fig = plt.figure(figsize=(15.0, 10.6), dpi=320)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.10, 1.0, 1.0], wspace=0.32, hspace=0.55)
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    ax_d = fig.add_subplot(grid[2, 0])
    ax_e = fig.add_subplot(grid[2, 1])

    plot_allocation_heatmap(ax_a, allocation)
    plot_allocation_shifts(
        ax_b,
        allocation,
        "global_li_max_benchmark",
        "Global",
        "economic_choice_baseline",
        "Global",
        "B. Shift from economic choice to Global Li-max",
        "#CC79A7",
        min_abs_value=0.5,
    )
    plot_allocation_shifts(
        ax_c,
        allocation,
        "domestic_li_allocation_objective",
        "China",
        "global_li_max_benchmark",
        "Global",
        "C. China allocation relative to Global Li-max",
        TARGET_COLORS["China"],
        min_abs_value=0.5,
    )
    plot_allocation_shifts(
        ax_d,
        allocation,
        "domestic_li_allocation_objective",
        "European Union",
        "global_li_max_benchmark",
        "Global",
        "D. EU allocation relative to Global Li-max",
        TARGET_COLORS["European Union"],
        min_abs_value=0.5,
    )
    plot_tradeoff(ax_e, summary)

    fig.suptitle("Country-scale effects of recovered lithium allocation objectives", fontsize=15, weight="bold", y=0.985)
    fig.text(
        0.5,
        0.035,
        "2050 reference-policy benchmark. Shifts are decomposed into the resource-efficiency move from economic choice to Global Li-max and the additional regional allocation shifts relative to Global Li-max.",
        ha="center",
        fontsize=9.5,
        color="#374151",
    )
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.10, top=0.92)

    output_png = FIGURE_DIR / "Figure_country_scale_allocation_effects.png"
    output_pdf = FIGURE_DIR / "Figure_country_scale_allocation_effects.pdf"
    fig.savefig(output_png, transparent=True)
    fig.savefig(output_pdf, transparent=True)
    plt.close(fig)
    print(f"Wrote {output_png}")
    print(f"Wrote {output_pdf}")


if __name__ == "__main__":
    main()
