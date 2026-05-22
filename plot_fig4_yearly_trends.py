from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "unified_policy_run" / "data" / "lithium_loss_scenarios_unified_yearly_parallel"
OUT_DIR = ROOT / "unified_policy_run" / "figures" / "fig4_yearly_trends"

POLICY_ORDER = [
    "current_policy",
    "reference_policy",
    "strict_policy",
    "critical_route_policy",
]

MITIGATION_ORDER = [
    "capacity_expansion",
    "policy_relaxation",
    "high_direct_maturity",
    "lithium_aware_high_price",
    "high_recovery_efficiency",
    "combined_mitigation",
    "max_lithium",
]

COLORS = {
    "capacity_expansion": "#8C8C8C",
    "policy_relaxation": "#4C78A8",
    "high_direct_maturity": "#F58518",
    "lithium_aware_high_price": "#B279A2",
    "high_recovery_efficiency": "#54A24B",
    "combined_mitigation": "#E45756",
    "max_lithium": "#111827",
}


def read_yearly():
    frames = []
    for path in sorted(BASE.glob("year_*/lithium_loss_scenarios_summary.csv")):
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No yearly summaries found under {BASE}")
    df = pd.concat(frames, ignore_index=True)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype(int)
    for col in [
        "total_lithium_loss_t",
        "loss_reduction_vs_baseline_t",
        "loss_reduction_vs_baseline_pct",
        "recovered_lithium_t",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DDE2E6", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_policy_panels(df):
    df = df[df["mitigation_scenario"].isin(MITIGATION_ORDER)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, policy in zip(axes, POLICY_ORDER):
        sub = df[df["policy_scenario"] == policy]
        for mitigation in MITIGATION_ORDER:
            series = sub[sub["mitigation_scenario"] == mitigation].sort_values("year")
            ax.plot(
                series["year"],
                series["loss_reduction_vs_baseline_pct"],
                label=mitigation.replace("_", " "),
                color=COLORS[mitigation],
                linewidth=2.0 if mitigation in {"combined_mitigation", "max_lithium"} else 1.5,
                alpha=0.95,
            )
        ax.set_title(policy.replace("_policy", "").replace("_", " ").title(), loc="left", weight="bold")
        ax.set_ylabel("Loss reduction vs baseline, %")
        style(ax)
    axes[-2].set_xlabel("Year")
    axes[-1].set_xlabel("Year")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Fig4 yearly mitigation effectiveness, 2025-2050", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0.10, 1, 0.95))
    out = OUT_DIR / "fig4_yearly_mitigation_loss_reduction_by_policy.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_average_panel(df):
    sub = df[df["mitigation_scenario"].isin(MITIGATION_ORDER)].copy()
    avg = (
        sub.groupby(["year", "mitigation_scenario"], as_index=False)
        ["loss_reduction_vs_baseline_pct"]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    for mitigation in MITIGATION_ORDER:
        series = avg[avg["mitigation_scenario"] == mitigation].sort_values("year")
        ax.plot(
            series["year"],
            series["loss_reduction_vs_baseline_pct"],
            label=mitigation.replace("_", " "),
            color=COLORS[mitigation],
            linewidth=2.4 if mitigation in {"combined_mitigation", "max_lithium"} else 1.8,
        )
    ax.set_title("Average across policy scenarios", loc="left", weight="bold")
    ax.set_ylabel("Loss reduction vs baseline, %")
    ax.set_xlabel("Year")
    style(ax)
    ax.legend(loc="upper left", ncol=2, frameon=False)
    fig.suptitle("Fig4 yearly mitigation effectiveness, policy average", fontsize=15, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT_DIR / "fig4_yearly_mitigation_loss_reduction_policy_average.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def write_rank_table(df):
    sub = df[df["mitigation_scenario"].isin(MITIGATION_ORDER)].copy()
    avg = (
        sub.groupby(["year", "mitigation_scenario"], as_index=False)
        ["loss_reduction_vs_baseline_pct"]
        .mean()
    )
    avg["rank"] = avg.groupby("year")["loss_reduction_vs_baseline_pct"].rank(
        ascending=False, method="dense"
    )
    return avg.sort_values(["year", "rank", "mitigation_scenario"])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    df = read_yearly()
    combined = OUT_DIR / "fig4_yearly_lithium_loss_scenarios_summary.csv"
    df.sort_values(["year", "policy_scenario", "mitigation_scenario"]).to_csv(combined, index=False)
    rank = write_rank_table(df)
    rank_path = OUT_DIR / "fig4_yearly_mitigation_rank_policy_average.csv"
    rank.to_csv(rank_path, index=False)
    outputs = [combined, rank_path, plot_policy_panels(df), plot_average_panel(df)]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
