from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "Figure_data" / "joint_policy_technology" / "regional_technology_choice"
OUT_DIR = DATA_DIR / "pyrohydro_comparison"
SHARE_FILE = DATA_DIR / "regional_technology_choice_shares.csv"

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
TECHNOLOGY_ORDER = ["Direct", "Hydro", "PyroHydro"]
TECH_COLORS = {
    "Direct": "#0072B2",
    "Hydro": "#009E73",
    "PyroHydro": "#D55E00",
}
POLICY_LINESTYLES = {
    "reference_policy": "-",
    "current_policy": "--",
    "strict_policy": "-.",
    "critical_route_policy": ":",
}
REGION_ORDER = ["Global", "CHN", "KOR", "IND", "USA"]
REGION_LABELS = {
    "Global": "Global",
    "CHN": "China",
    "KOR": "Korea",
    "IND": "India",
    "USA": "USA",
}


def load_shares():
    data = pd.read_csv(SHARE_FILE)
    for column in ["year", "technology_share_pct"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data[
        data["region"].isin(REGION_ORDER)
        & data["policy_scenario"].isin(POLICY_ORDER)
        & data["technology"].isin(TECHNOLOGY_ORDER)
        & data["year"].between(2030, 2050)
    ].copy()
    return data


def plot_global_by_policy(data):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), dpi=300, sharey=True)
    global_data = data[data["region"] == "Global"].copy()
    for ax, policy in zip(axes, POLICY_ORDER):
        subset = global_data[global_data["policy_scenario"] == policy]
        for technology in TECHNOLOGY_ORDER:
            line = subset[subset["technology"] == technology].sort_values("year")
            ax.plot(
                line["year"],
                line["technology_share_pct"],
                color=TECH_COLORS[technology],
                linewidth=2.2,
                label=technology,
            )
        ax.set_title(POLICY_LABELS[policy], fontsize=11, fontweight="bold")
        ax.set_xlim(2030, 2050)
        ax.set_ylim(0, 100)
        ax.set_xticks([2030, 2035, 2040, 2045, 2050])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color="0.9")
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.0)
    axes[0].set_ylabel("Technology share (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle("Global technology mix after adding PyroHydro", fontsize=13)
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    fig.savefig(OUT_DIR / "pyrohydro_global_policy_trend.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "pyrohydro_global_policy_trend.pdf", bbox_inches="tight")


def plot_regions_by_policy(data):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), dpi=300, sharex=True, sharey=True)
    for ax, region in zip(axes.flat, REGION_ORDER):
        subset_region = data[data["region"] == region]
        for technology in TECHNOLOGY_ORDER:
            for policy in POLICY_ORDER:
                line = subset_region[
                    (subset_region["technology"] == technology)
                    & (subset_region["policy_scenario"] == policy)
                ].sort_values("year")
                ax.plot(
                    line["year"],
                    line["technology_share_pct"],
                    color=TECH_COLORS[technology],
                    linestyle=POLICY_LINESTYLES[policy],
                    linewidth=2.0 if policy == "reference_policy" else 1.4,
                    alpha=0.95 if policy == "reference_policy" else 0.7,
                )
        ax.set_title(REGION_LABELS[region], fontsize=12, fontweight="bold")
        ax.set_xlim(2030, 2050)
        ax.set_ylim(0, 100)
        ax.set_xticks([2030, 2035, 2040, 2045, 2050])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y", color="0.9")
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.0)
    axes.flat[-1].axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("Technology share (%)")
    for ax in axes[-1, :2]:
        ax.set_xlabel("Year")

    tech_handles = [
        plt.Line2D([0], [0], color=TECH_COLORS[tech], lw=2.2, label=tech)
        for tech in TECHNOLOGY_ORDER
    ]
    policy_handles = [
        plt.Line2D(
            [0],
            [0],
            color="0.25",
            lw=1.8,
            linestyle=POLICY_LINESTYLES[policy],
            label=POLICY_LABELS[policy],
        )
        for policy in POLICY_ORDER
    ]
    fig.legend(
        handles=tech_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.33, 0.01),
        title="Technology",
    )
    fig.legend(
        handles=policy_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.72, 0.01),
        title="Policy",
    )
    fig.suptitle("Regional technology mix after adding PyroHydro", fontsize=14)
    fig.tight_layout(rect=[0, 0.09, 1, 0.94])
    fig.savefig(OUT_DIR / "pyrohydro_selected_regions_policy_trend.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / "pyrohydro_selected_regions_policy_trend.pdf", bbox_inches="tight")


def plot_annual_dynamic_style(data):
    global_data = data[data["region"] == "Global"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    for policy in POLICY_ORDER:
        for technology in TECHNOLOGY_ORDER:
            subset = global_data[
                (global_data["policy_scenario"] == policy)
                & (global_data["technology"] == technology)
            ].sort_values("year")
            axes[0].plot(
                subset["year"],
                subset["technology_share_pct"],
                color=TECH_COLORS[technology],
                linestyle=POLICY_LINESTYLES[policy],
                linewidth=2.1 if policy == "reference_policy" else 1.5,
                alpha=0.95 if policy == "reference_policy" else 0.75,
            )
            axes[1].plot(
                subset["year"],
                subset["technology_throughput_t"] / 1_000_000,
                color=TECH_COLORS[technology],
                linestyle=POLICY_LINESTYLES[policy],
                linewidth=2.1 if policy == "reference_policy" else 1.5,
                alpha=0.95 if policy == "reference_policy" else 0.75,
            )

    axes[0].set_title("Global technology share by policy scenario")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Technology share (%)")
    axes[0].set_xlim(2025, 2050)
    axes[0].set_xticks([2025, 2030, 2035, 2040, 2045, 2050])
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis="y", color="0.9")

    axes[1].set_title("Global allocated throughput by technology")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Allocated scrap throughput (Mt)")
    axes[1].set_xlim(2025, 2050)
    axes[1].set_xticks([2025, 2030, 2035, 2040, 2045, 2050])
    axes[1].grid(axis="y", color="0.9")

    technology_handles = [
        plt.Line2D([0], [0], color=TECH_COLORS[tech], lw=2.2, label=tech)
        for tech in TECHNOLOGY_ORDER
    ]
    policy_handles = [
        plt.Line2D(
            [0],
            [0],
            color="0.25",
            lw=1.8,
            linestyle=POLICY_LINESTYLES[policy],
            label=POLICY_LABELS[policy],
        )
        for policy in POLICY_ORDER
    ]
    fig.legend(
        handles=technology_handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.38, 0.95),
        title="Technology (line color)",
    )
    fig.legend(
        handles=policy_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.67, 0.82),
        title="Policy scenario (line style)",
    )
    fig.suptitle(
        "Annual technology mix after adding PyroHydro choice layer",
        y=1.03,
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.84])
    fig.savefig(
        OUT_DIR / "annual_dynamic_direct_share_trend_pyrohydro.png",
        bbox_inches="tight",
    )
    fig.savefig(
        OUT_DIR / "annual_dynamic_direct_share_trend_pyrohydro.pdf",
        bbox_inches="tight",
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_shares()
    data.to_csv(OUT_DIR / "pyrohydro_global_selected_region_shares.csv", index=False)
    plot_global_by_policy(data)
    plot_regions_by_policy(data)
    plot_annual_dynamic_style(data)
    print(f"Wrote {OUT_DIR / 'pyrohydro_global_policy_trend.png'}")
    print(f"Wrote {OUT_DIR / 'pyrohydro_selected_regions_policy_trend.png'}")
    print(f"Wrote {OUT_DIR / 'annual_dynamic_direct_share_trend_pyrohydro.png'}")
    print(f"Wrote {OUT_DIR / 'pyrohydro_global_selected_region_shares.csv'}")


if __name__ == "__main__":
    main()
