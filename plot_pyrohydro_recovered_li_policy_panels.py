import argparse
from pathlib import Path

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE_DIR = ROOT / "Figure_data" / "joint_policy_technology"
SCENARIOS = {
    "conservative": BASE_DIR / "pyrohydro_sensitivity_conservative_annual_gurobi",
    "medium": BASE_DIR / "pyrohydro_sensitivity_medium_annual_gurobi",
    "strong": BASE_DIR / "pyrohydro_sensitivity_strong",
}
SCENARIO_LABELS = {
    "conservative": "Conservative PyroHydro",
    "medium": "Medium PyroHydro",
    "strong": "Strong PyroHydro substitution",
}
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
TECHNOLOGY_ORDER = ["Direct", "Hydro", "Pyro", "PyroHydro"]
TECHNOLOGY_COLORS = {
    "Direct": "#0072B2",
    "Hydro": "#009E73",
    "Pyro": "#D55E00",
    "PyroHydro": "#CC79A7",
}
PANEL_LABELS = ["a", "b", "c", "d"]


def read_scenario_data(data_dir):
    corrected_path = data_dir / "annual_dynamic_trend_corrected.csv"
    source_path = corrected_path if corrected_path.exists() else data_dir / "dynamic_scale_summary.csv"
    data = pd.read_csv(source_path)
    for column in ["year", "recovered_lithium_t"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def plot_scenario(name, data_dir, out_dir):
    data = read_scenario_data(data_dir)
    data = data[data["technology"].isin(TECHNOLOGY_ORDER)].copy()
    technologies = [
        technology
        for technology in TECHNOLOGY_ORDER
        if technology in set(data["technology"].dropna())
    ]
    ymax = max(float(data["recovered_lithium_t"].max()) / 1000.0 * 1.12, 1.0)

    plt.rcParams.update({"font.family": "Arial"})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharex=True, sharey=True, dpi=300)
    axes = axes.ravel()

    for ax, policy, panel_label in zip(axes, POLICY_ORDER, PANEL_LABELS):
        subset = data[data["policy_scenario"] == policy].copy()
        for technology in technologies:
            line = subset[subset["technology"] == technology].sort_values("year")
            ax.plot(
                line["year"],
                line["recovered_lithium_t"] / 1000.0,
                color=TECHNOLOGY_COLORS[technology],
                linewidth=2.4,
                label=technology,
            )
        ax.text(
            0.02,
            0.95,
            panel_label,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            va="top",
        )
        ax.set_title(POLICY_LABELS[policy], fontsize=12, pad=8)
        ax.set_xlim(2025, 2050)
        ax.set_ylim(0, ymax)
        ax.set_xticks([2025, 2030, 2035, 2040, 2045, 2050])
        ax.grid(axis="y", color="0.88", linewidth=0.8)

    for ax in axes[2:]:
        ax.set_xlabel("Year")
    for ax in axes[::2]:
        ax.set_ylabel("Recovered Li (kt)")

    handles = [
        Line2D([0], [0], color=TECHNOLOGY_COLORS[technology], lw=2.6, label=technology)
        for technology in technologies
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
        title="Technology",
    )
    fig.suptitle(
        f"Recovered lithium by technology and policy scenario: {SCENARIO_LABELS[name]}",
        y=1.015,
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.91])

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}_recovered_li_by_policy_technology.png"
    pdf = out_dir / f"{name}_recovered_li_by_policy_technology.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenarios",
        default="conservative,medium,strong",
        help="Comma-separated scenario keys: conservative, medium, strong.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(BASE_DIR / "pyrohydro_recovered_li_policy_panels"),
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    for name in [item.strip() for item in args.scenarios.split(",") if item.strip()]:
        if name not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {name}")
        plot_scenario(name, SCENARIOS[name], out_dir)


if __name__ == "__main__":
    main()
