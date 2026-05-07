from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RATE_DETAIL_FILE = (
    ROOT / "Scenario result" / "recycling_rate" / "lithium_recycling_rate_detail.csv"
)
SCENARIO_SUMMARY_FILE = (
    ROOT
    / "Figure_data"
    / "joint_policy_technology"
    / "lithium_loss_scenarios"
    / "lithium_loss_scenarios_summary.csv"
)
OUTPUT_DIR = ROOT / "Figure_data" / "joint_policy_technology"
YEARS = [2030, 2040, 2050]
COLLECTION_SCENARIOS = ["low_collection", "baseline", "high_collection"]
SCENARIO_LABELS = {
    "low_collection": "Low collection",
    "baseline": "Baseline",
    "high_collection": "High collection",
}
SCENARIO_COLORS = {
    "low_collection": "#E15759",
    "baseline": "#4E79A7",
    "high_collection": "#59A14F",
}
KEY_COUNTRIES = [
    "China",
    "USA",
    "India",
    "Germany",
    "Japan",
    "Korea",
]


def load_capture_data():
    data = pd.read_csv(RATE_DETAIL_FILE)
    # Capture fields are independent of recycling method and recovery path; keep one slice
    # to avoid counting the same retired lithium three times.
    data = data[
        (data["recovery_efficiency_scenario"] == "baseline")
        & (data["recycling_m"] == "Hydro")
        & (data["Year"].isin(YEARS))
        & (data["scenario"].isin(COLLECTION_SCENARIOS))
    ].copy()
    return data


def global_capture_summary(data):
    summary = (
        data.groupby(["Year", "scenario"], as_index=False)[
            ["retired_lithium", "collected_lithium", "uncollected_lithium"]
        ]
        .sum()
        .rename(
            columns={
                "retired_lithium": "retired_li_t",
                "collected_lithium": "collected_li_t",
                "uncollected_lithium": "capture_loss_t",
            }
        )
    )
    summary["capture_loss_kt"] = summary["capture_loss_t"] / 1000.0
    summary["collection_rate_weighted"] = (
        summary["collected_li_t"] / summary["retired_li_t"]
    )
    return summary


def country_capture_summary(data):
    summary = (
        data.groupby(["Year", "scenario", "country"], as_index=False)[
            ["retired_lithium", "collected_lithium", "uncollected_lithium"]
        ]
        .sum()
        .rename(
            columns={
                "retired_lithium": "retired_li_t",
                "collected_lithium": "collected_li_t",
                "uncollected_lithium": "capture_loss_t",
            }
        )
    )
    summary["capture_loss_kt"] = summary["capture_loss_t"] / 1000.0
    summary["collection_rate_weighted"] = np.where(
        summary["retired_li_t"] > 0,
        summary["collected_li_t"] / summary["retired_li_t"],
        np.nan,
    )
    return summary


def load_technology_loss():
    summary = pd.read_csv(SCENARIO_SUMMARY_FILE)
    tech = summary[
        (summary["mitigation_scenario"] == "baseline")
        & (summary["policy_scenario"] == "reference_policy")
        & (summary["year"].isin(YEARS))
    ].copy()
    tech["technology_pathway_loss_kt"] = (
        tech["technology_recovery_loss_t"] / 1000.0
    )
    return tech[["year", "technology_pathway_loss_kt"]]


def plot_global_capture(ax, global_summary):
    x = np.arange(len(YEARS))
    width = 0.24
    offsets = [-width, 0, width]
    for offset, scenario in zip(offsets, COLLECTION_SCENARIOS):
        subset = global_summary[global_summary["scenario"] == scenario].set_index("Year")
        values = [subset.loc[year, "capture_loss_kt"] for year in YEARS]
        ax.bar(
            x + offset,
            values,
            width=width,
            color=SCENARIO_COLORS[scenario],
            label=SCENARIO_LABELS[scenario],
            edgecolor="white",
            linewidth=0.6,
        )
    ax.set_title("A. Capture loss under collection-rate scenarios", loc="left", fontsize=11, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(year) for year in YEARS])
    ax.set_ylabel("Capture loss (kt Li)")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def plot_capture_vs_technology(ax, global_summary, tech):
    high = global_summary[global_summary["scenario"] == "high_collection"].set_index("Year")
    base = global_summary[global_summary["scenario"] == "baseline"].set_index("Year")
    low = global_summary[global_summary["scenario"] == "low_collection"].set_index("Year")
    tech = tech.set_index("year")
    x = np.arange(len(YEARS))
    capture_base = np.array([base.loc[year, "capture_loss_kt"] for year in YEARS])
    capture_low = np.array([low.loc[year, "capture_loss_kt"] for year in YEARS])
    capture_high = np.array([high.loc[year, "capture_loss_kt"] for year in YEARS])
    tech_values = np.array([tech.loc[year, "technology_pathway_loss_kt"] for year in YEARS])
    ax.plot(x, capture_base, marker="o", color="#4E79A7", linewidth=2, label="Capture loss baseline")
    ax.fill_between(
        x,
        capture_high,
        capture_low,
        color="#4E79A7",
        alpha=0.16,
        label="Collection scenario range",
    )
    ax.plot(x, tech_values, marker="s", color="#F28E2B", linewidth=2, label="Technology pathway loss")
    ax.set_title("B. Capture loss range vs technology pathway loss", loc="left", fontsize=11, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(year) for year in YEARS])
    ax.set_ylabel("Loss (kt Li)")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def plot_top_countries(ax, country_summary):
    focus = country_summary[
        (country_summary["Year"] == 2050) & (country_summary["scenario"] == "baseline")
    ].copy()
    top = focus.sort_values("capture_loss_kt", ascending=False).head(10)
    top = top.sort_values("capture_loss_kt")
    ax.barh(
        top["country"],
        top["capture_loss_kt"],
        color="#9C755F",
        edgecolor="white",
        linewidth=0.6,
    )
    for y_pos, value in enumerate(top["capture_loss_kt"]):
        ax.text(value + max(top["capture_loss_kt"]) * 0.015, y_pos, f"{value:.1f}", va="center", fontsize=8)
    ax.set_title("C. Largest country contributors to capture loss\n(2050 baseline collection)", loc="left", fontsize=11, weight="bold")
    ax.set_xlabel("Capture loss (kt Li)")
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_key_rates(ax, country_summary):
    focus = country_summary[
        (country_summary["Year"] == 2050)
        & (country_summary["country"].isin(KEY_COUNTRIES))
    ].copy()
    table = focus.pivot_table(
        index="country",
        columns="scenario",
        values="collection_rate_weighted",
        aggfunc="mean",
    ).reindex(KEY_COUNTRIES)
    x = np.arange(len(KEY_COUNTRIES))
    width = 0.24
    offsets = [-width, 0, width]
    for offset, scenario in zip(offsets, COLLECTION_SCENARIOS):
        values = table[scenario].values * 100.0
        ax.bar(
            x + offset,
            values,
            width=width,
            color=SCENARIO_COLORS[scenario],
            edgecolor="white",
            linewidth=0.6,
            label=SCENARIO_LABELS[scenario],
        )
    ax.set_title("D. Key-country collection-rate assumptions\n(2050)", loc="left", fontsize=11, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(KEY_COUNTRIES, rotation=25, ha="right")
    ax.set_ylabel("Li-weighted collection rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    plt.rcParams["font.family"] = "Arial"
    capture_data = load_capture_data()
    global_summary = global_capture_summary(capture_data)
    country_summary = country_capture_summary(capture_data)
    tech = load_technology_loss()

    global_summary.to_csv(OUTPUT_DIR / "capture_loss_global_sensitivity.csv", index=False)
    country_summary.to_csv(OUTPUT_DIR / "capture_loss_country_sensitivity.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0), dpi=300)
    plot_global_capture(axes[0, 0], global_summary)
    plot_capture_vs_technology(axes[0, 1], global_summary, tech)
    plot_top_countries(axes[1, 0], country_summary)
    plot_key_rates(axes[1, 1], country_summary)
    fig.suptitle(
        "Capture-loss uncertainty under collection-rate assumptions",
        fontsize=15,
        weight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.015,
        "Capture loss is modeled as retired lithium that is not collected. Scenario ranges reflect collection-rate assumptions rather than observed historical loss.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    png = OUTPUT_DIR / "capture_loss_sensitivity_figure.png"
    pdf = OUTPUT_DIR / "capture_loss_sensitivity_figure.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
