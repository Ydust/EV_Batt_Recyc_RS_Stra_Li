from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "Figure_data" / "joint_policy_technology" / "lithium_loss_scenarios"
SUMMARY_FILE = OUTPUT_DIR / "lithium_loss_scenarios_summary.csv"
PARETO_FILES = {
    2030: OUTPUT_DIR / "lithium_pareto_frontier_2030.csv",
    2040: OUTPUT_DIR / "lithium_pareto_frontier_2040.csv",
    2050: OUTPUT_DIR / "lithium_pareto_frontier.csv",
}
FIGURE_DIR = ROOT / "Figure_data" / "joint_policy_technology"

YEARS = [2030, 2040, 2050]
MAIN_SCENARIOS = ["baseline", "combined_mitigation", "max_lithium"]
SCENARIO_LABELS = {
    "baseline": "Baseline",
    "combined_mitigation": "Combined",
    "max_lithium": "Max-Li",
}
POLICIES = ["reference_policy", "strict_policy", "critical_route_policy"]
POLICY_LABELS = {
    "reference_policy": "Reference",
    "strict_policy": "Strict",
    "critical_route_policy": "Critical-route",
}
SCENARIO_COLORS = {
    "baseline": "#B8DBB3",
    "combined_mitigation": "#E29135",
    "max_lithium": "#CC79A7",
}
YEAR_COLORS = {
    2030: "#719AAC",
    2040: "#72B063",
    2050: "#E29135",
}
POLICY_MARKERS = {
    "reference_policy": "o",
    "strict_policy": "s",
    "critical_route_policy": "^",
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


def load_pareto():
    frames = []
    for year, path in PARETO_FILES.items():
        data = pd.read_csv(path)
        data["year"] = year
        frames.append(data)
    return pd.concat(frames, ignore_index=True)


def plot_recovered_by_year(ax, summary):
    data = summary[
        (summary["year"].isin(YEARS))
        & (summary["mitigation_scenario"].isin(MAIN_SCENARIOS))
        & (summary["policy_scenario"] == "reference_policy")
    ].copy()
    table = (
        data.pivot_table(
            index="year",
            columns="mitigation_scenario",
            values="recovered_lithium_t",
            aggfunc="mean",
        )
        .reindex(YEARS)[MAIN_SCENARIOS]
        / 1000.0
    )
    x = np.arange(len(YEARS))
    width = 0.22
    offsets = np.linspace(-width, width, len(MAIN_SCENARIOS))
    for offset, scenario in zip(offsets, MAIN_SCENARIOS):
        values = table[scenario].to_numpy()
        ax.bar(
            x + offset,
            values,
            width=width,
            color=SCENARIO_COLORS[scenario],
            edgecolor="black",
            linewidth=0.5,
            label=SCENARIO_LABELS[scenario],
        )
        for xi, value in zip(x + offset, values):
            ax.text(xi, value + max(table.max()) * 0.012, f"{value:,.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(YEARS)
    ax.set_ylabel("Recovered Li (kt Li)", fontsize=10)
    ax.set_title("A. Max-Li is a tight upper-bound on recovered Li", loc="left", fontsize=11, weight="bold")
    ax.set_ylim(0, table.max().max() * 1.16)
    format_axes(ax)


def plot_remaining_gap(ax, summary):
    data = summary[
        (summary["year"].isin(YEARS))
        & (summary["mitigation_scenario"].isin(["combined_mitigation", "max_lithium"]))
        & (summary["policy_scenario"] == "reference_policy")
    ].copy()
    table = (
        data.pivot_table(
            index="year",
            columns="mitigation_scenario",
            values="recovered_lithium_t",
            aggfunc="mean",
        )
        .reindex(YEARS)
        / 1000.0
    )
    gain = table["max_lithium"] - table["combined_mitigation"]
    bars = ax.bar(
        np.arange(len(YEARS)),
        gain.to_numpy(),
        color=[YEAR_COLORS[year] for year in YEARS],
        edgecolor="black",
        linewidth=0.5,
    )
    for bar, value in zip(bars, gain.to_numpy()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(gain.max() * 0.08, 0.02),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_xticks(np.arange(len(YEARS)))
    ax.set_xticklabels(YEARS)
    ax.set_ylabel("Max-Li gain over combined (kt Li)", fontsize=10)
    ax.set_title("B. Remaining optimization headroom is small", loc="left", fontsize=11, weight="bold")
    ax.set_ylim(0, gain.max() * 1.30)
    format_axes(ax)


def plot_pareto_reference(ax, pareto):
    for year in YEARS:
        data = pareto[
            (pareto["year"] == year)
            & (pareto["policy_scenario"] == "reference_policy")
        ].sort_values("frontier_step")
        x = (data["route_modeled_cost"] - data["route_modeled_cost"].iloc[0]) / 1e6
        y = (data["recovered_lithium_t"] - data["recovered_lithium_t"].iloc[0]) / 1000.0
        ax.plot(
            x,
            y,
            color=YEAR_COLORS[year],
            marker="o",
            linewidth=2.0,
            markersize=4.5,
            label=str(year),
        )
    ax.set_xlabel("Additional modeled system cost (million USD)", fontsize=10)
    ax.set_ylabel("Additional recovered Li (kt Li)", fontsize=10)
    ax.set_title("C. Pareto frontier: extra Li requires additional cost", loc="left", fontsize=11, weight="bold")
    format_axes(ax, grid_axis="both")


def plot_cost_to_max_li(ax, pareto):
    rows = []
    for (year, policy), group in pareto.groupby(["year", "policy_scenario"]):
        if policy not in POLICIES:
            continue
        group = group.sort_values("frontier_step")
        first = group.iloc[0]
        last = group.iloc[-1]
        rows.append(
            {
                "year": year,
                "policy_scenario": policy,
                "cost_increase_m": (last["route_modeled_cost"] - first["route_modeled_cost"]) / 1e6,
            }
        )
    data = pd.DataFrame(rows)
    table = data.pivot_table(index="year", columns="policy_scenario", values="cost_increase_m").reindex(YEARS)[POLICIES]
    x = np.arange(len(YEARS))
    width = 0.22
    offsets = np.linspace(-width, width, len(POLICIES))
    for offset, policy in zip(offsets, POLICIES):
        values = table[policy].to_numpy()
        ax.bar(
            x + offset,
            values,
            width=width,
            color="#FFFFFF",
            edgecolor=YEAR_COLORS[2050] if policy == "strict_policy" else "#111827",
            linewidth=1.0,
            hatch={"reference_policy": "", "strict_policy": "///", "critical_route_policy": "..."}[policy],
            label=POLICY_LABELS[policy],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(YEARS)
    ax.set_ylabel("Cost increase to Max-Li (million USD)", fontsize=10)
    ax.set_title("D. Policy constraints mainly raise the cost of reaching Max-Li", loc="left", fontsize=11, weight="bold")
    format_axes(ax)


def main():
    plt.rcParams["font.family"] = "Arial"
    summary = pd.read_csv(SUMMARY_FILE)
    pareto = load_pareto()

    fig = plt.figure(figsize=(13.5, 8.2), dpi=320)
    grid = fig.add_gridspec(2, 2, wspace=0.27, hspace=0.38)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    plot_recovered_by_year(ax_a, summary)
    plot_remaining_gap(ax_b, summary)
    plot_pareto_reference(ax_c, pareto)
    plot_cost_to_max_li(ax_d, pareto)

    handles_a, labels_a = ax_a.get_legend_handles_labels()
    handles_c, labels_c = ax_c.get_legend_handles_labels()
    handles_d, labels_d = ax_d.get_legend_handles_labels()
    fig.legend(
        handles_a + handles_c + handles_d,
        labels_a + labels_c + labels_d,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Max-Li benchmark and cost-recovery Pareto frontier",
        fontsize=15,
        weight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.035,
        "Advanced technology and capacity scenarios. Pareto curves show additional cost required to move from cost-minimizing recovery toward Max-Li.",
        ha="center",
        fontsize=9.5,
        color="#374151",
    )
    fig.subplots_adjust(left=0.075, right=0.97, bottom=0.10, top=0.86, wspace=0.28, hspace=0.40)

    output_png = FIGURE_DIR / "Figure_max_li_pareto_frontier.png"
    output_pdf = FIGURE_DIR / "Figure_max_li_pareto_frontier.pdf"
    fig.savefig(output_png, bbox_inches="tight", transparent=True)
    fig.savefig(output_pdf, bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"Wrote {output_png}")
    print(f"Wrote {output_pdf}")


if __name__ == "__main__":
    main()
