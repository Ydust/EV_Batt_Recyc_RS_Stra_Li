from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = (
    ROOT
    / "unified_policy_run"
    / "data"
    / "lithium_loss_scenarios_yearly_sensitivity"
    / "_figures"
    / "sensitivity_summary_all.csv"
)
OUT_DIR = (
    ROOT
    / "unified_policy_run"
    / "figures"
    / "fig4_yearly_sensitivity_comparison"
)

SCENARIO_ORDER = [
    "avail_thr_0.4",
    "avail_thr_0.8",
    "direct_mult_1.0",
    "direct_mult_1.5",
    "li_price_3",
    "li_price_5",
    "penalty_150",
    "penalty_600",
    "delay_5",
    "delay_10",
]

SCENARIO_LABELS = {
    "avail_thr_0.4": "Avail. threshold 0.4",
    "avail_thr_0.8": "Avail. threshold 0.8",
    "direct_mult_1.0": "Direct cost x1.0",
    "direct_mult_1.5": "Direct cost x1.5",
    "li_price_3": "Li price x3",
    "li_price_5": "Li price x5",
    "penalty_150": "Policy penalty 150",
    "penalty_600": "Policy penalty 600",
    "delay_5": "Delay cost 5",
    "delay_10": "Delay cost 10",
}

COLORS = {
    "avail_thr_0.4": "#4C78A8",
    "avail_thr_0.8": "#72B7B2",
    "direct_mult_1.0": "#F58518",
    "direct_mult_1.5": "#E45756",
    "li_price_3": "#B279A2",
    "li_price_5": "#9D755D",
    "penalty_150": "#54A24B",
    "penalty_600": "#2F6B3F",
    "delay_5": "#EECA3B",
    "delay_10": "#B6992D",
}


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DDE2E6", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)


def read_data():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source file: {SOURCE}")
    df = pd.read_csv(SOURCE)
    numeric_cols = [
        "year",
        "total_lithium_loss_t",
        "loss_reduction_vs_baseline_pct",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["sensitivity_scenario"].isin(SCENARIO_ORDER)].copy()
    df["scenario_label"] = df["sensitivity_scenario"].map(SCENARIO_LABELS)
    return df


def plot_comparison(df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

    baseline = (
        df[
            (df["mitigation_scenario"] == "baseline")
            & (df["policy_scenario"] == "current_policy")
        ]
        .groupby(["sensitivity_scenario", "year"], as_index=False)[
            "total_lithium_loss_t"
        ]
        .mean()
    )
    for scenario in SCENARIO_ORDER:
        series = baseline[baseline["sensitivity_scenario"] == scenario].sort_values(
            "year"
        )
        axes[0].plot(
            series["year"],
            series["total_lithium_loss_t"] / 1_000,
            label=SCENARIO_LABELS[scenario],
            color=COLORS[scenario],
            linewidth=2.0,
            alpha=0.92,
        )
    axes[0].set_title(
        "Baseline lithium loss under current policy",
        loc="left",
        fontsize=12,
        weight="bold",
    )
    axes[0].set_ylabel("Total lithium loss, kt Li")
    axes[0].set_xlabel("Year")
    style(axes[0])

    key = df[
        df["mitigation_scenario"].isin(["combined_mitigation", "max_lithium"])
    ].copy()
    avg = (
        key.groupby(["sensitivity_scenario", "year", "mitigation_scenario"], as_index=False)[
            "loss_reduction_vs_baseline_pct"
        ]
        .mean()
    )
    line_styles = {"combined_mitigation": "-", "max_lithium": "--"}
    for scenario in SCENARIO_ORDER:
        for mitigation in ["combined_mitigation", "max_lithium"]:
            series = avg[
                (avg["sensitivity_scenario"] == scenario)
                & (avg["mitigation_scenario"] == mitigation)
            ].sort_values("year")
            axes[1].plot(
                series["year"],
                series["loss_reduction_vs_baseline_pct"],
                color=COLORS[scenario],
                linestyle=line_styles[mitigation],
                linewidth=2.0 if mitigation == "combined_mitigation" else 1.6,
                alpha=0.9,
            )
    axes[1].set_title(
        "Mitigation loss reduction, policy average",
        loc="left",
        fontsize=12,
        weight="bold",
    )
    axes[1].set_ylabel("Loss reduction vs baseline, %")
    axes[1].set_xlabel("Year")
    style(axes[1])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    axes[1].plot([], [], color="#2F3437", linestyle="-", label="combined mitigation")
    axes[1].plot([], [], color="#2F3437", linestyle="--", label="max lithium")
    axes[1].legend(loc="upper left", frameon=False)

    fig.suptitle(
        "Fig4 yearly sensitivity comparison, 2025-2050",
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0.14, 1, 0.93))
    out = OUT_DIR / "fig4_yearly_sensitivity_comparison.png"
    fig.savefig(out, dpi=240)
    plt.close(fig)
    return out


def plot_heatmap(df):
    sub = df[df["mitigation_scenario"] == "combined_mitigation"].copy()
    avg = (
        sub.groupby(["sensitivity_scenario", "year"], as_index=False)[
            "loss_reduction_vs_baseline_pct"
        ]
        .mean()
    )
    heat = avg.pivot(
        index="sensitivity_scenario",
        columns="year",
        values="loss_reduction_vs_baseline_pct",
    ).reindex(SCENARIO_ORDER)

    fig, ax = plt.subplots(figsize=(13, 5.8))
    im = ax.imshow(heat.values, aspect="auto", cmap="YlGnBu", vmin=30, vmax=85)
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels([SCENARIO_LABELS[item] for item in heat.index])
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels([str(int(year)) for year in heat.columns], rotation=45, ha="right")
    ax.set_xlabel("Year")
    ax.set_title(
        "Combined mitigation loss reduction, policy average",
        loc="left",
        fontsize=12,
        weight="bold",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Loss reduction vs baseline, %")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.suptitle(
        "Fig4 yearly sensitivity heatmap, 2025-2050",
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = OUT_DIR / "fig4_yearly_sensitivity_combined_mitigation_heatmap.png"
    fig.savefig(out, dpi=240)
    plt.close(fig)
    return out


def write_summary(df):
    summary = (
        df[
            (df["year"].isin([2030, 2040, 2050]))
            & (
                df["mitigation_scenario"].isin(
                    ["baseline", "combined_mitigation", "max_lithium"]
                )
            )
        ]
        .groupby(
            [
                "sensitivity_scenario",
                "scenario_label",
                "year",
                "policy_scenario",
                "mitigation_scenario",
            ],
            as_index=False,
        )
        .agg(
            total_lithium_loss_t=("total_lithium_loss_t", "mean"),
            loss_reduction_vs_baseline_pct=(
                "loss_reduction_vs_baseline_pct",
                "mean",
            ),
        )
    )
    out = OUT_DIR / "fig4_yearly_sensitivity_comparison_summary.csv"
    summary.to_csv(out, index=False)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    df = read_data()
    outputs = [write_summary(df), plot_comparison(df), plot_heatmap(df)]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
