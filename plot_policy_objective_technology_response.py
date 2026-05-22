from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "Figure_data" / "joint_policy_technology" / "policy_objective_technology_response"
FIGURE_DIR = ROOT / "Figure_data" / "joint_policy_technology"
MIX_FILE = INPUT_DIR / "policy_objective_technology_response_mix.csv"
SUMMARY_FILE = INPUT_DIR / "policy_objective_technology_response_summary.csv"

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
TARGETS = ["China", "United States", "European Union"]
TARGET_LABELS = {
    "China": "China",
    "United States": "US",
    "European Union": "EU",
}
TECHNOLOGIES = ["Direct", "Hydro", "Pyro"]
TECH_COLORS = {
    "Direct": "#719AAC",
    "Hydro": "#72B063",
    "Pyro": "#E29135",
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


def policy_target_table(mix, technology):
    data = mix[
        (mix["objective"] == "domestic_li_allocation_objective")
        & (mix["scope"] == "target_region")
        & (mix["technology"] == technology)
        & (mix["target_region"].isin(TARGETS))
    ].copy()
    table = data.pivot_table(
        index="policy_scenario",
        columns="target_region",
        values="technology_share_pct",
        fill_value=0.0,
    )
    return table.reindex(index=POLICIES, columns=TARGETS, fill_value=0.0)


def plot_direct_heatmap(ax, mix):
    table = policy_target_table(mix, "Direct")
    image = ax.imshow(table.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_xticks(np.arange(len(TARGETS)))
    ax.set_xticklabels([TARGET_LABELS[target] for target in TARGETS])
    ax.set_yticks(np.arange(len(POLICIES)))
    ax.set_yticklabels([POLICY_LABELS[policy] for policy in POLICIES])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("black")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = table.values[i, j]
            color = "white" if value > 58 else "black"
            ax.text(j, i, f"{value:.0f}%", ha="center", va="center", fontsize=9, color=color)
    cbar = plt.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label("Direct share of target-region Li (%)", fontsize=9)
    ax.set_title("A. Regional Li-access objectives lock target-region recovery into Direct", loc="left", fontsize=11, weight="bold")


def plot_stacked_target_mix(ax, mix):
    data = mix[
        (mix["objective"] == "domestic_li_allocation_objective")
        & (mix["scope"] == "target_region")
        & (mix["target_region"].isin(TARGETS))
    ].copy()
    x_labels = []
    x = []
    bottoms = []
    cursor = 0
    for target in TARGETS:
        for policy in POLICIES:
            x.append(cursor)
            x_labels.append(f"{TARGET_LABELS[target]}\n{POLICY_LABELS[policy]}")
            bottoms.append(0.0)
            cursor += 1
        cursor += 0.7

    bottom = np.zeros(len(x))
    for technology in TECHNOLOGIES:
        values = []
        for target in TARGETS:
            for policy in POLICIES:
                row = data[
                    (data["target_region"] == target)
                    & (data["policy_scenario"] == policy)
                    & (data["technology"] == technology)
                ]
                values.append(float(row["technology_share_pct"].iloc[0]) if not row.empty else 0.0)
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=TECH_COLORS[technology],
            edgecolor="black",
            linewidth=0.35,
            label=technology,
        )
        bottom += np.array(values)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7.8)
    ax.set_ylabel("Technology share of target-region Li (%)", fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_title("B. Target-region technology mix is invariant across policy settings", loc="left", fontsize=11, weight="bold")
    format_axes(ax)


def plot_direct_delta(ax, mix):
    table = policy_target_table(mix, "Direct")
    current = table.loc["current_policy"]
    delta = table.subtract(current, axis="columns").drop(index="current_policy")
    y_labels = []
    values = []
    colors = []
    for policy in ["reference_policy", "strict_policy", "critical_route_policy"]:
        for target in TARGETS:
            y_labels.append(f"{POLICY_LABELS[policy]} - {TARGET_LABELS[target]}")
            value = float(delta.loc[policy, target])
            values.append(value)
            colors.append("#719AAC" if value >= 0 else "#D9D9D9")
    order = np.argsort(values)
    y = np.arange(len(values))
    ordered_values = np.array(values)[order]
    ordered_labels = np.array(y_labels)[order]
    ordered_colors = np.array(colors)[order]
    ax.barh(y, ordered_values, color=ordered_colors, edgecolor="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=1.0)
    for yi, value in zip(y, ordered_values):
        ha = "left" if value >= 0 else "right"
        offset = 1.2 if value >= 0 else -1.2
        ax.text(value + offset, yi, f"{value:+.1f} pp", va="center", ha=ha, fontsize=8.4)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered_labels, fontsize=8.5)
    ax.set_xlabel("Change in Direct share vs Current policy (percentage points)", fontsize=9.5)
    ax.set_title("C. Direct-share response is effectively zero", loc="left", fontsize=11, weight="bold")
    max_abs = max(5.0, float(np.nanmax(np.abs(ordered_values))) * 1.35)
    ax.set_xlim(-max_abs, max_abs)
    format_axes(ax, grid_axis="x")


def plot_cost_penalty(ax, summary):
    data = summary[
        (summary["objective"] == "domestic_li_allocation_objective")
        & (summary["target_region"].isin(TARGETS))
    ].copy()
    current = data[data["policy_scenario"] == "current_policy"].set_index("target_region")[
        "route_modeled_cost"
    ]
    rows = []
    for _, row in data.iterrows():
        target = row["target_region"]
        rows.append(
            {
                "policy_scenario": row["policy_scenario"],
                "target_region": target,
                "cost_delta_musd": (row["route_modeled_cost"] - current[target]) / 1e6,
            }
        )
    table = pd.DataFrame(rows).pivot_table(
        index="policy_scenario",
        columns="target_region",
        values="cost_delta_musd",
        fill_value=0.0,
    ).reindex(index=POLICIES, columns=TARGETS, fill_value=0.0)
    x = np.arange(len(TARGETS))
    width = 0.2
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(POLICIES))
    for offset, policy in zip(offsets, POLICIES):
        values = table.loc[policy].to_numpy()
        ax.bar(
            x + offset,
            values,
            width=width,
            color="#FFFFFF",
            edgecolor="#111827",
            linewidth=0.8,
            hatch={"reference_policy": "", "current_policy": "//", "strict_policy": "xx", "critical_route_policy": "..."}[policy],
            label=POLICY_LABELS[policy],
        )
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS[target] for target in TARGETS])
    ax.set_ylabel("Cost change vs Current policy (million USD)", fontsize=10)
    ax.set_title("D. Policy constraints appear as regional cost penalties instead", loc="left", fontsize=11, weight="bold")
    max_abs = max(1.0, float(np.nanmax(np.abs(table.values))) * 1.25)
    ax.set_ylim(-max_abs, max_abs)
    format_axes(ax)


def main():
    plt.rcParams["font.family"] = "Arial"
    mix = pd.read_csv(MIX_FILE)
    summary = pd.read_csv(SUMMARY_FILE)

    fig = plt.figure(figsize=(15.2, 9.0), dpi=320)
    grid = fig.add_gridspec(2, 2, wspace=0.31, hspace=0.42)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    plot_direct_heatmap(ax_a, mix)
    plot_stacked_target_mix(ax_b, mix)
    plot_direct_delta(ax_c, mix)
    plot_cost_penalty(ax_d, summary)

    handles_b, labels_b = ax_b.get_legend_handles_labels()
    handles_d, labels_d = ax_d.get_legend_handles_labels()
    fig.legend(
        handles_b + handles_d,
        labels_b + labels_d,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=7,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Regional secondary-lithium access objectives saturate Direct technology choice",
        fontsize=15,
        weight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.035,
        "2050 high-collection, advanced-capacity benchmark. Under hard regional Li-access maximization, technology mix has little elasticity; policy constraints mainly change regional access and modeled system cost.",
        ha="center",
        fontsize=9.5,
        color="#374151",
    )
    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.12, top=0.86)

    output_png = FIGURE_DIR / "Figure3_policy_objective_technology_response.png"
    output_pdf = FIGURE_DIR / "Figure3_policy_objective_technology_response.pdf"
    fig.savefig(output_png, transparent=True)
    fig.savefig(output_pdf, transparent=True)
    plt.close(fig)
    print(f"Wrote {output_png}")
    print(f"Wrote {output_pdf}")


if __name__ == "__main__":
    main()
