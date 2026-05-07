import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "Figure_data" / "policy_constraint_strength" / "policy_constraint_strength_summary.csv"
OUTPUT_DIR = ROOT / "Figure_data" / "policy_constraint_strength"

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
COLORS = {
    "reference_policy": "#4C78A8",
    "current_policy": "#F58518",
    "strict_policy": "#B279A2",
    "critical_route_policy": "#E45756",
}


def plot_policy_strength(data, method, output_dir):
    method_data = data[data["method"] == method].copy()
    if method_data.empty:
        raise ValueError(f"No rows for method: {method}")

    method_data["policy_scenario"] = pd.Categorical(
        method_data["policy_scenario"], POLICY_ORDER, ordered=True
    )
    method_data = method_data.sort_values(["year", "policy_scenario"])

    years = sorted(method_data["year"].unique())
    x = range(len(years))
    width = 0.18
    offsets = {
        "reference_policy": -1.5 * width,
        "current_policy": -0.5 * width,
        "strict_policy": 0.5 * width,
        "critical_route_policy": 1.5 * width,
    }

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8), sharex=True)
    for policy in POLICY_ORDER:
        subset = method_data[method_data["policy_scenario"] == policy].set_index("year")
        if subset.empty:
            continue
        xpos = [i + offsets[policy] for i in x]
        axes[0].bar(
            xpos,
            subset.loc[years, "cross_border_scrap_t"] / 1e6,
            width=width,
            label=POLICY_LABELS[policy],
            color=COLORS[policy],
        )
        axes[1].bar(
            xpos,
            subset.loc[years, "policy_cost_delta"] / 1e6,
            width=width,
            label=POLICY_LABELS[policy],
            color=COLORS[policy],
        )

    axes[0].set_ylabel("Cross-border scrap\nmillion tonnes")
    axes[1].set_ylabel("Policy cost delta\nmillion cost units")
    axes[1].set_xlabel("Year")
    axes[0].set_title(
        f"Policy constraint strength comparison ({method}, Strategy 3)",
        fontsize=14,
        weight="bold",
    )
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels([str(year) for year in years])
    axes[0].legend(ncol=3, frameon=False, loc="upper left")

    fig.tight_layout()
    output = output_dir / f"policy_constraint_strength_{method}.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)

    diag = method_data.pivot_table(
        index="year",
        columns="policy_scenario",
        values=[
            "forbidden_route_count",
            "penalized_route_count",
            "used_forbidden_flow_t",
        ],
        aggfunc="first",
    )
    diag.to_csv(output_dir / f"policy_constraint_strength_{method}_diagnostics.csv")

    fig, ax1 = plt.subplots(figsize=(10.5, 4.8))
    for policy in POLICY_ORDER:
        subset = method_data[method_data["policy_scenario"] == policy].set_index("year")
        ax1.plot(
            years,
            subset.loc[years, "forbidden_route_count"],
            marker="o",
            label=f"{POLICY_LABELS[policy]} forbidden routes",
            color=COLORS[policy],
        )
    ax1.set_ylabel("Forbidden route count")
    ax1.set_xlabel("Year")
    ax1.grid(axis="y", alpha=0.25)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_title(
        f"Forbidden-route diagnostics ({method}, Strategy 3)",
        fontsize=14,
        weight="bold",
    )
    ax2 = ax1.twinx()
    strict = method_data[method_data["policy_scenario"] == "strict_policy"].set_index("year")
    critical = method_data[
        method_data["policy_scenario"] == "critical_route_policy"
    ].set_index("year")
    ax2.plot(
        years,
        strict.loc[years, "used_forbidden_flow_t"],
        color="#111827",
        linestyle="--",
        marker="s",
        label="Used forbidden flow",
    )
    ax2.set_ylabel("Used forbidden flow (t)")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, ncol=2, frameon=False, loc="upper left")
    fig.tight_layout()
    diag_png = output_dir / f"policy_constraint_strength_{method}_diagnostics.png"
    fig.savefig(diag_png, dpi=300)
    plt.close(fig)
    return output, diag_png


def plot_policy_strength_delta(data, method, output_dir):
    method_data = data[data["method"] == method].copy()
    method_data["policy_scenario"] = pd.Categorical(
        method_data["policy_scenario"], POLICY_ORDER, ordered=True
    )
    years = sorted(method_data["year"].unique())
    baseline = method_data[method_data["policy_scenario"] == "current_policy"].set_index("year")
    strict = method_data[method_data["policy_scenario"] == "strict_policy"].set_index("year")
    critical = method_data[
        method_data["policy_scenario"] == "critical_route_policy"
    ].set_index("year")
    open_policy = method_data[method_data["policy_scenario"] == "reference_policy"].set_index("year")

    delta = pd.DataFrame(
        {
            "year": years,
            "current_cost_increase_vs_reference": (
                baseline.loc[years, "policy_cost_delta"]
                - open_policy.loc[years, "policy_cost_delta"]
            ).values,
            "strict_cost_increase_vs_reference": (
                strict.loc[years, "policy_cost_delta"]
                - open_policy.loc[years, "policy_cost_delta"]
            ).values,
            "critical_cost_increase_vs_reference": (
                critical.loc[years, "policy_cost_delta"]
                - open_policy.loc[years, "policy_cost_delta"]
            ).values
            if not critical.empty
            else [0] * len(years),
            "current_cross_border_change_vs_reference": (
                baseline.loc[years, "cross_border_scrap_t"]
                - open_policy.loc[years, "cross_border_scrap_t"]
            ).values,
            "strict_cross_border_change_vs_reference": (
                strict.loc[years, "cross_border_scrap_t"]
                - open_policy.loc[years, "cross_border_scrap_t"]
            ).values,
            "critical_cross_border_change_vs_reference": (
                critical.loc[years, "cross_border_scrap_t"]
                - open_policy.loc[years, "cross_border_scrap_t"]
            ).values
            if not critical.empty
            else [0] * len(years),
            "strict_extra_forbidden_routes_vs_current": (
                strict.loc[years, "forbidden_route_count"]
                - baseline.loc[years, "forbidden_route_count"]
            ).values,
            "strict_used_forbidden_flow_t": strict.loc[
                years, "used_forbidden_flow_t"
            ].values,
            "critical_used_forbidden_flow_t": critical.loc[
                years, "used_forbidden_flow_t"
            ].values
            if not critical.empty
            else [0] * len(years),
        }
    )
    delta.to_csv(output_dir / f"policy_constraint_strength_{method}_delta_vs_reference.csv", index=False)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14, 4.8),
        gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]},
    )
    x = range(len(years))
    width = 0.35
    axes[0].bar(
        [i - width / 2 for i in x],
        delta["current_cost_increase_vs_reference"] / 1e6,
        width=width,
        color=COLORS["current_policy"],
        label="Current vs Reference",
    )
    axes[0].plot(
        x,
        delta["critical_cost_increase_vs_reference"] / 1e6,
        color=COLORS["critical_route_policy"],
        marker="s",
        linewidth=2,
        label="Critical-route vs Reference",
    )
    axes[0].bar(
        [i + width / 2 for i in x],
        delta["strict_cost_increase_vs_reference"] / 1e6,
        width=width,
        color=COLORS["strict_policy"],
        label="Strict vs Reference",
    )
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels([str(year) for year in years], rotation=30)
    axes[0].set_ylabel("Additional policy cost\nmillion cost units")
    axes[0].set_title("Cost effect")
    axes[0].legend(frameon=False)

    axes[1].axhline(0, color="#111827", linewidth=1)
    axes[1].plot(
        years,
        delta["current_cross_border_change_vs_reference"] / 1e6,
        marker="o",
        color=COLORS["current_policy"],
        label="Current",
    )
    axes[1].plot(
        years,
        delta["critical_cross_border_change_vs_reference"] / 1e6,
        marker="s",
        color=COLORS["critical_route_policy"],
        label="Critical-route",
    )
    axes[1].plot(
        years,
        delta["strict_cross_border_change_vs_reference"] / 1e6,
        marker="o",
        color=COLORS["strict_policy"],
        label="Strict",
    )
    axes[1].set_ylabel("Cross-border flow change\nmillion tonnes")
    axes[1].set_title("Path reallocation")
    axes[1].set_xticks(years)
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].bar(
        years,
        delta["strict_extra_forbidden_routes_vs_current"],
        color=COLORS["strict_policy"],
        width=3.2,
        label="Extra forbidden routes",
    )
    axes[2].plot(
        years,
        delta["strict_used_forbidden_flow_t"],
        color="#111827",
        marker="s",
        linestyle="--",
        label="Used forbidden flow",
    )
    axes[2].plot(
        years,
        delta["critical_used_forbidden_flow_t"],
        color=COLORS["critical_route_policy"],
        marker="D",
        linestyle="-",
        label="Critical used forbidden flow",
    )
    axes[2].set_title("Blocking diagnosis")
    axes[2].set_ylabel("Count / tonnes")
    axes[2].set_xticks(years)
    axes[2].tick_params(axis="x", rotation=30)
    axes[2].legend(frameon=False)

    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Policy constraint strength: effects relative to reference policy ({method}, Strategy 3)",
        fontsize=15,
        weight="bold",
    )
    fig.text(
        0.5,
        0.01,
        "Interpretation: in the current parameterization, stricter policy increases compliance cost but does not reroute or block optimized flows.",
        ha="center",
        fontsize=10,
        color="#374151",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.92])
    output = output_dir / f"policy_constraint_strength_{method}_delta_vs_reference.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def main():
    parser = argparse.ArgumentParser(description="Plot policy constraint strength results.")
    parser.add_argument("--input", default=str(INPUT))
    parser.add_argument("--method", default="Direct")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = plot_policy_strength(data, args.method, output_dir)
    delta_output = plot_policy_strength_delta(data, args.method, output_dir)
    for output in outputs:
        print(f"Wrote {output}")
    print(f"Wrote {delta_output}")


if __name__ == "__main__":
    main()
