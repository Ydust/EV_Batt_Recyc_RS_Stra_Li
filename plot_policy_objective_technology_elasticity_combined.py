from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE_DIR = ROOT / "Figure_data" / "joint_policy_technology" / "policy_objective_technology_elasticity"
FIGURE_DIR = ROOT / "Figure_data" / "joint_policy_technology"
CHINA_MIX = BASE_DIR / "policy_objective_technology_elasticity_mix.csv"
CHINA_SUMMARY = BASE_DIR / "policy_objective_technology_elasticity_summary.csv"
US_EU_MIX = BASE_DIR / "aggregate_us_eu" / "policy_objective_technology_elasticity_mix.csv"
US_EU_SUMMARY = BASE_DIR / "aggregate_us_eu" / "policy_objective_technology_elasticity_summary.csv"

YEARS = [2030, 2040, 2050]
POLICIES = ["reference_policy", "current_policy", "strict_policy", "critical_route_policy"]
POLICY_LABELS = {
    "reference_policy": "Reference",
    "current_policy": "Current",
    "strict_policy": "Strict",
    "critical_route_policy": "Critical-route",
}
TARGETS = ["China", "United States", "European Union"]
TARGET_LABELS = {"China": "China", "United States": "US", "European Union": "EU"}
TECHNOLOGIES = ["Direct", "Hydro", "Pyro"]
TECH_COLORS = {"Direct": "#719AAC", "Hydro": "#72B063", "Pyro": "#E29135"}
YEAR_COLORS = {2030: "#719AAC", 2040: "#E29135", 2050: "#2E8B2E"}


def load_combined():
    mix = pd.concat([pd.read_csv(CHINA_MIX), pd.read_csv(US_EU_MIX)], ignore_index=True)
    summary = pd.concat([pd.read_csv(CHINA_SUMMARY), pd.read_csv(US_EU_SUMMARY)], ignore_index=True)
    return mix, summary


def format_axes(ax, grid_axis="y"):
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9, direction="in")
    ax.grid(axis=grid_axis, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)


def direct_table(mix, target_share, policy):
    data = mix[
        (mix["scope"] == "target_region")
        & (mix["technology"] == "Direct")
        & (mix["policy_scenario"] == policy)
        & (np.isclose(mix["target_share_of_max"], target_share))
        & (mix["target_region"].isin(TARGETS))
    ].copy()
    table = data.pivot_table(
        index="target_region",
        columns="year",
        values="technology_share_pct",
        fill_value=0.0,
    )
    return table.reindex(index=TARGETS, columns=YEARS, fill_value=0.0)


def plot_heatmap(ax, mix, target_share, policy, title):
    table = direct_table(mix, target_share, policy)
    image = ax.imshow(table.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(YEARS)))
    ax.set_xticklabels(YEARS)
    ax.set_yticks(np.arange(len(TARGETS)))
    ax.set_yticklabels([TARGET_LABELS[target] for target in TARGETS])
    ax.tick_params(length=0)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.values[i, j]
            ax.text(
                j,
                i,
                f"{value:.0f}%",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value > 58 else "black",
            )
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")
    cbar = plt.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Direct share of target-region Li (%)", fontsize=9)
    ax.set_title(title, loc="left", fontsize=11, weight="bold")


def plot_frontier(ax, mix, target_region, policy):
    data = mix[
        (mix["scope"] == "target_region")
        & (mix["technology"] == "Direct")
        & (mix["target_region"] == target_region)
        & (mix["policy_scenario"] == policy)
    ].copy()
    for year in YEARS:
        subset = data[data["year"] == year].sort_values("target_share_of_max")
        ax.plot(
            subset["target_share_of_max"] * 100,
            subset["technology_share_pct"],
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            color=YEAR_COLORS[year],
            label=str(year),
        )
    ax.set_xlabel("Required target-region Li access (% of max)", fontsize=10)
    ax.set_ylabel("Direct share of target-region Li (%)", fontsize=10)
    ax.set_ylim(-2, 102)
    ax.set_title(
        f"C. {TARGET_LABELS[target_region]} Direct frontier under {POLICY_LABELS[policy]} policy",
        loc="left",
        fontsize=11,
        weight="bold",
    )
    format_axes(ax, grid_axis="both")


def plot_cost_at_max(ax, summary, policy):
    data = summary[
        (summary["target_region"].isin(TARGETS))
        & (summary["policy_scenario"] == policy)
        & (np.isclose(summary["target_share_of_max"], 1.0))
    ].copy()
    reference = summary[
        (summary["target_region"].isin(TARGETS))
        & (summary["policy_scenario"] == policy)
        & (np.isclose(summary["target_share_of_max"], 0.0))
    ][["year", "target_region", "route_modeled_cost"]].rename(
        columns={"route_modeled_cost": "cost_at_zero"}
    )
    data = data.merge(reference, on=["year", "target_region"], how="left")
    data["cost_delta_musd"] = (data["route_modeled_cost"] - data["cost_at_zero"]) / 1e6
    x = np.arange(len(TARGETS))
    width = 0.22
    offsets = np.linspace(-width, width, len(YEARS))
    for offset, year in zip(offsets, YEARS):
        values = [
            float(
                data[
                    (data["target_region"] == target)
                    & (data["year"] == year)
                ]["cost_delta_musd"].iloc[0]
            )
            for target in TARGETS
        ]
        ax.bar(
            x + offset,
            values,
            width=width,
            color=YEAR_COLORS[year],
            edgecolor="black",
            linewidth=0.45,
            label=str(year),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS[target] for target in TARGETS])
    ax.set_ylabel("Cost increase to 100% access target (million USD)", fontsize=10)
    ax.set_title(
        f"D. Cost of forcing the Direct-dominated upper bound ({POLICY_LABELS[policy]})",
        loc="left",
        fontsize=11,
        weight="bold",
    )
    format_axes(ax)


def main():
    plt.rcParams["font.family"] = "Arial"
    mix, summary = load_combined()

    fig = plt.figure(figsize=(15.2, 9.0), dpi=320)
    grid = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.40)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    plot_heatmap(
        ax_a,
        mix,
        0.95,
        "strict_policy",
        "A. 95% access target remains Hydro-dominated across regions",
    )
    plot_heatmap(
        ax_b,
        mix,
        1.0,
        "strict_policy",
        "B. 100% access target triggers Direct-dominated solutions",
    )
    plot_frontier(ax_c, mix, "European Union", "strict_policy")
    plot_cost_at_max(ax_d, summary, "strict_policy")

    handles_c, labels_c = ax_c.get_legend_handles_labels()
    handles_d, labels_d = ax_d.get_legend_handles_labels()
    seen = set()
    handles = []
    labels = []
    for handle, label in list(zip(handles_c, labels_c)) + list(zip(handles_d, labels_d)):
        if label not in seen:
            handles.append(handle)
            labels.append(label)
            seen.add(label)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Technology substitution thresholds for regional secondary-lithium access",
        fontsize=15,
        weight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.035,
        "China, US, and EU aggregate results under strict policy. Soft regional access targets show a threshold response: Hydro satisfies most sub-maximum access requirements, while Direct appears at the upper bound.",
        ha="center",
        fontsize=9.4,
        color="#374151",
    )
    fig.subplots_adjust(left=0.075, right=0.965, bottom=0.11, top=0.86)

    output_png = FIGURE_DIR / "Figure3_policy_objective_technology_elasticity_combined.png"
    output_pdf = FIGURE_DIR / "Figure3_policy_objective_technology_elasticity_combined.pdf"
    fig.savefig(output_png, transparent=True)
    fig.savefig(output_pdf, transparent=True)
    plt.close(fig)
    print(f"Wrote {output_png}")
    print(f"Wrote {output_pdf}")


if __name__ == "__main__":
    main()
