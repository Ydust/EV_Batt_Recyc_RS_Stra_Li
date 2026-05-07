from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "Figure_data" / "joint_policy_technology" / "policy_objective_technology_elasticity"
FIGURE_DIR = ROOT / "Figure_data" / "joint_policy_technology"
MIX_FILE = INPUT_DIR / "policy_objective_technology_elasticity_mix.csv"
SUMMARY_FILE = INPUT_DIR / "policy_objective_technology_elasticity_summary.csv"

POLICIES = ["reference_policy", "current_policy", "strict_policy", "critical_route_policy"]
YEARS = [2030, 2040, 2050]
POLICY_LABELS = {
    "reference_policy": "Reference",
    "current_policy": "Current",
    "strict_policy": "Strict",
    "critical_route_policy": "Critical-route",
}
TARGETS = ["China"]
TARGET_LABELS = {"China": "China"}
TECHNOLOGIES = ["Direct", "Hydro", "Pyro"]
TECH_COLORS = {"Direct": "#719AAC", "Hydro": "#72B063", "Pyro": "#E29135"}


def format_axes(ax, grid_axis="y"):
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=9, direction="in")
    ax.grid(axis=grid_axis, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)


def direct_table(mix, target_share):
    data = mix[
        (mix["scope"] == "target_region")
        & (mix["technology"] == "Direct")
        & (mix["target_region"] == "China")
        & (np.isclose(mix["target_share_of_max"], target_share))
    ]
    table = data.pivot_table(index="policy_scenario", columns="year", values="technology_share_pct", fill_value=0)
    return table.reindex(index=POLICIES, columns=YEARS, fill_value=0)


def plot_direct_heatmap(ax, mix):
    table = direct_table(mix, 0.95)
    image = ax.imshow(table.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(YEARS)))
    ax.set_xticklabels(YEARS)
    ax.set_yticks(np.arange(len(POLICIES)))
    ax.set_yticklabels([POLICY_LABELS[p] for p in POLICIES])
    ax.tick_params(length=0)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.values[i, j]
            ax.text(j, i, f"{value:.0f}%", ha="center", va="center", fontsize=9, color="white" if value > 58 else "black")
    cbar = plt.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Direct share at 95% China access (%)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title("A. Direct substitution at 95% China access over time", loc="left", fontsize=11, weight="bold")


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
            label=str(year),
        )
    ax.set_xlabel("Required target-region Li access (% of max)", fontsize=10)
    ax.set_ylabel("Direct share of target-region Li (%)", fontsize=10)
    ax.set_ylim(-2, 102)
    ax.set_title(f"B. {TARGET_LABELS[target_region]} Direct frontier over time ({POLICY_LABELS[policy]})", loc="left", fontsize=11, weight="bold")
    format_axes(ax, grid_axis="both")


def plot_stacked_mix(ax, mix):
    data = mix[
        (mix["scope"] == "target_region")
        & (mix["target_region"] == "China")
        & (np.isclose(mix["target_share_of_max"], 0.95))
        & (mix["policy_scenario"].isin(POLICIES))
    ].copy()
    x = []
    labels = []
    cursor = 0
    for year in YEARS:
        for policy in POLICIES:
            x.append(cursor)
            labels.append(f"{year}\n{POLICY_LABELS[policy]}")
            cursor += 1
        cursor += 0.7
    bottom = np.zeros(len(x))
    for tech in TECHNOLOGIES:
        values = []
        for year in YEARS:
            for policy in POLICIES:
                row = data[
                    (data["year"] == year)
                    & (data["policy_scenario"] == policy)
                    & (data["technology"] == tech)
                ]
                values.append(float(row["technology_share_pct"].iloc[0]) if not row.empty else 0.0)
        ax.bar(x, values, bottom=bottom, color=TECH_COLORS[tech], edgecolor="black", linewidth=0.35, label=tech)
        bottom += np.array(values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.8)
    ax.set_ylabel("Technology share of target-region Li (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title("C. China technology mix at 95% access target over time", loc="left", fontsize=11, weight="bold")
    format_axes(ax)


def plot_cost_frontier(ax, summary):
    data = summary[summary["target_region"] == "China"].copy()
    for year in YEARS:
        ref = data[
            (data["year"] == year)
            & (data["policy_scenario"] == "reference_policy")
            & (np.isclose(data["target_share_of_max"], 0.0))
        ]["route_modeled_cost"].iloc[0]
        subset = data[
            (data["year"] == year)
            & (data["policy_scenario"] == "strict_policy")
        ].sort_values("target_share_of_max")
        ax.plot(
            subset["target_share_of_max"] * 100,
            (subset["route_modeled_cost"] - ref) / 1e6,
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            label=str(year),
        )
    ax.set_xlabel("Required target-region Li access (% of max)", fontsize=10)
    ax.set_ylabel("Cost penalty under strict policy (million USD)", fontsize=10)
    ax.set_title("D. Cost of China access targets rises over time under strict policy", loc="left", fontsize=11, weight="bold")
    format_axes(ax, grid_axis="both")


def main():
    plt.rcParams["font.family"] = "Arial"
    mix = pd.read_csv(MIX_FILE)
    summary = pd.read_csv(SUMMARY_FILE)

    fig = plt.figure(figsize=(15.2, 9.0), dpi=320)
    grid = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.42)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])
    plot_direct_heatmap(ax_a, mix)
    plot_frontier(ax_b, mix, "China", "strict_policy")
    plot_stacked_mix(ax_c, mix)
    plot_cost_frontier(ax_d, summary)

    handles_c, labels_c = ax_c.get_legend_handles_labels()
    handles_b, labels_b = ax_b.get_legend_handles_labels()
    handles_d, labels_d = ax_d.get_legend_handles_labels()
    fig.legend(handles_c + handles_b + handles_d, labels_c + labels_b + labels_d, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=10, frameon=False, fontsize=8.5)
    fig.suptitle("Policy-constrained technology substitution under soft regional Li-access targets", fontsize=15, weight="bold", y=0.985)
    fig.text(0.5, 0.035, "2050 high-collection, advanced-capacity benchmark. Regional Li access is imposed as a fractional target, allowing the model to trade off Direct/Hydro/Pyro selection against cost and policy-constrained routes.", ha="center", fontsize=9.5, color="#374151")
    fig.subplots_adjust(left=0.075, right=0.965, bottom=0.12, top=0.86)

    output_png = FIGURE_DIR / "Figure3_policy_objective_technology_elasticity.png"
    output_pdf = FIGURE_DIR / "Figure3_policy_objective_technology_elasticity.pdf"
    fig.savefig(output_png, transparent=True)
    fig.savefig(output_pdf, transparent=True)
    plt.close(fig)
    print(f"Wrote {output_png}")
    print(f"Wrote {output_pdf}")


if __name__ == "__main__":
    main()
