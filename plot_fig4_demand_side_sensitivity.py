from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "unified_policy_run" / "data" / "fig4_demand_side_sensitivity"
OUT_DIR = ROOT / "unified_policy_run" / "figures" / "fig4_demand_side_sensitivity"

SUPPLY_CASES = ["eol_only", "eol_plus_manufacturing"]
COLLECTIONS = ["low_collection", "baseline", "high_collection"]
RECOVERIES = ["low", "baseline", "high"]
KEY_YEARS = [2030, 2040, 2050]
POLICIES = [
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
MITIGATION_LABELS = {
    "capacity_expansion": "Capacity\nexpansion",
    "policy_relaxation": "Policy\nrelaxation",
    "high_direct_maturity": "Direct\nmaturity",
    "lithium_aware_high_price": "Li-value\naware",
    "high_recovery_efficiency": "High\nrecovery",
    "combined_mitigation": "Combined",
    "max_lithium": "Max Li",
}

SUPPLY_LABELS = {
    "eol_only": "EOL only",
    "eol_plus_manufacturing": "EOL + manufacturing",
}
COLLECTION_LABELS = {
    "low_collection": "Low collection",
    "baseline": "Baseline collection",
    "high_collection": "High collection",
}
RECOVERY_LABELS = {
    "low": "Low recovery",
    "baseline": "Baseline recovery",
    "high": "High recovery",
}


def read_results():
    frames = []
    missing = []
    for supply_case in SUPPLY_CASES:
        for collection in COLLECTIONS:
            for recovery in RECOVERIES:
                path = (
                    BASE
                    / supply_case
                    / f"collection_{collection}"
                    / f"recovery_{recovery}"
                    / "lithium_loss_scenarios_summary.csv"
                )
                if not path.exists():
                    missing.append(path)
                    continue
                df = pd.read_csv(path)
                df["supply_case"] = supply_case
                df["collection_scenario"] = collection
                df["recovery_scenario"] = recovery
                frames.append(df)
    if missing:
        raise FileNotFoundError(
            "Missing demand-side sensitivity summaries:\n"
            + "\n".join(str(path) for path in missing)
        )
    data = pd.concat(frames, ignore_index=True)
    for col in [
        "year",
        "total_lithium_loss_t",
        "loss_reduction_vs_baseline_t",
        "loss_reduction_vs_baseline_pct",
    ]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["supply_label"] = data["supply_case"].map(SUPPLY_LABELS)
    data["collection_label"] = data["collection_scenario"].map(COLLECTION_LABELS)
    data["recovery_label"] = data["recovery_scenario"].map(RECOVERY_LABELS)
    data["demand_case_label"] = (
        data["supply_label"]
        + " | "
        + data["collection_label"]
        + " | "
        + data["recovery_label"]
    )
    return data


def style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DDE2E6", linewidth=0.8)
    ax.set_axisbelow(True)


def write_key_tables(data):
    key = data[
        data["year"].isin(KEY_YEARS)
        & data["mitigation_scenario"].isin(
            ["baseline", "combined_mitigation", "max_lithium"]
        )
    ].copy()
    ordered_cols = [
        "supply_case",
        "collection_scenario",
        "recovery_scenario",
        "policy_scenario",
        "mitigation_scenario",
        "year",
        "total_lithium_loss_t",
        "loss_reduction_vs_baseline_t",
        "loss_reduction_vs_baseline_pct",
    ]
    key_path = OUT_DIR / "fig4_demand_side_sensitivity_key_years.csv"
    key[ordered_cols].sort_values(ordered_cols[:6]).to_csv(key_path, index=False)

    spread = (
        key.groupby(["year", "policy_scenario", "mitigation_scenario"], as_index=False)
        .agg(
            total_loss_min_t=("total_lithium_loss_t", "min"),
            total_loss_median_t=("total_lithium_loss_t", "median"),
            total_loss_max_t=("total_lithium_loss_t", "max"),
            reduction_min_pct=("loss_reduction_vs_baseline_pct", "min"),
            reduction_median_pct=("loss_reduction_vs_baseline_pct", "median"),
            reduction_max_pct=("loss_reduction_vs_baseline_pct", "max"),
        )
        .sort_values(["year", "policy_scenario", "mitigation_scenario"])
    )
    spread_path = OUT_DIR / "fig4_demand_side_sensitivity_spread.csv"
    spread.to_csv(spread_path, index=False)

    ranking = (
        data[
            data["year"].isin(KEY_YEARS)
            & data["mitigation_scenario"].isin(MITIGATION_ORDER)
        ]
        .groupby(["year", "policy_scenario", "mitigation_scenario"], as_index=False)
        .agg(
            reduction_median_pct=("loss_reduction_vs_baseline_pct", "median"),
            reduction_min_pct=("loss_reduction_vs_baseline_pct", "min"),
            reduction_max_pct=("loss_reduction_vs_baseline_pct", "max"),
            total_loss_median_t=("total_lithium_loss_t", "median"),
        )
        .sort_values(
            ["year", "policy_scenario", "reduction_median_pct"],
            ascending=[True, True, False],
        )
    )
    ranking["rank"] = ranking.groupby(["year", "policy_scenario"])[
        "reduction_median_pct"
    ].rank(ascending=False, method="dense")
    ranking_path = OUT_DIR / "fig4_demand_side_sensitivity_mitigation_ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    return key_path, spread_path, ranking_path


def plot_2050_heatmap(data, policy_scenario="current_policy"):
    sub = data[
        (data["year"] == 2050)
        & (data["policy_scenario"] == policy_scenario)
        & (data["mitigation_scenario"] == "combined_mitigation")
    ].copy()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.5, 5.2),
        sharey=True,
        constrained_layout=True,
    )
    vmin = sub["loss_reduction_vs_baseline_pct"].min()
    vmax = sub["loss_reduction_vs_baseline_pct"].max()
    for ax, supply_case in zip(axes, SUPPLY_CASES):
        part = sub[sub["supply_case"] == supply_case]
        heat = (
            part.pivot(
                index="recovery_scenario",
                columns="collection_scenario",
                values="loss_reduction_vs_baseline_pct",
            )
            .reindex(index=RECOVERIES, columns=COLLECTIONS)
        )
        im = ax.imshow(heat.values, cmap="YlGnBu", vmin=vmin, vmax=vmax)
        ax.set_title(SUPPLY_LABELS[supply_case], loc="left", weight="bold")
        ax.set_xticks(range(len(COLLECTIONS)))
        ax.set_xticklabels(
            [COLLECTION_LABELS[item] for item in COLLECTIONS],
            rotation=25,
            ha="right",
            fontsize=9,
        )
        ax.set_yticks(range(len(RECOVERIES)))
        ax.set_yticklabels([RECOVERY_LABELS[item] for item in RECOVERIES])
        for i in range(len(RECOVERIES)):
            for j in range(len(COLLECTIONS)):
                value = heat.values[i, j]
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=9)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Loss reduction vs baseline, %")
    fig.suptitle(
        "Fig4 demand-side sensitivity: "
        f"{policy_scenario.replace('_policy', '').replace('_', ' ')} policy, combined mitigation in 2050",
        fontsize=14,
        weight="bold",
    )
    out = OUT_DIR / "fig4_demand_side_sensitivity_2050_heatmap.png"
    if policy_scenario != "current_policy":
        policy_slug = policy_scenario.replace("_policy", "").replace("_", "-")
        out = OUT_DIR / f"fig4_demand_side_sensitivity_2050_heatmap_{policy_slug}.png"
    fig.savefig(out, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_spread(data):
    sub = data[
        data["year"].isin(KEY_YEARS)
        & data["policy_scenario"].isin(POLICIES)
        & data["mitigation_scenario"].isin(["combined_mitigation", "max_lithium"])
    ].copy()
    spread = (
        sub.groupby(["year", "policy_scenario", "mitigation_scenario"], as_index=False)[
            "loss_reduction_vs_baseline_pct"
        ]
        .agg(["min", "median", "max"])
        .reset_index()
    )

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), sharey=True)
    axes = axes.ravel()
    colors = {"combined_mitigation": "#E45756", "max_lithium": "#111827"}
    offsets = {"combined_mitigation": -0.12, "max_lithium": 0.12}
    for ax, policy in zip(axes, POLICIES):
        part = spread[spread["policy_scenario"] == policy]
        for mitigation in ["combined_mitigation", "max_lithium"]:
            series = part[part["mitigation_scenario"] == mitigation].sort_values("year")
            x = np.arange(len(KEY_YEARS)) + offsets[mitigation]
            lower = series["median"] - series["min"]
            upper = series["max"] - series["median"]
            ax.errorbar(
                x,
                series["median"],
                yerr=[lower, upper],
                fmt="o",
                capsize=4,
                linewidth=1.8,
                markersize=5,
                color=colors[mitigation],
                label=mitigation.replace("_", " "),
            )
        ax.set_title(policy.replace("_policy", "").replace("_", " ").title(), loc="left", weight="bold")
        ax.set_xticks(range(len(KEY_YEARS)))
        ax.set_xticklabels([str(year) for year in KEY_YEARS])
        ax.set_xlabel("Year")
        ax.set_ylabel("Loss reduction vs baseline, %")
        style(ax)
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle(
        "Fig4 demand-side sensitivity spread across 18 cases",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = OUT_DIR / "fig4_demand_side_sensitivity_spread.png"
    fig.savefig(out, dpi=240)
    plt.close(fig)
    return out


def plot_all_mitigation_2050(data):
    sub = data[
        (data["year"] == 2050)
        & data["mitigation_scenario"].isin(MITIGATION_ORDER)
        & data["policy_scenario"].isin(POLICIES)
    ].copy()

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharey=True)
    axes = axes.ravel()
    rng = np.random.default_rng(7)
    for ax, policy in zip(axes, POLICIES):
        part = sub[sub["policy_scenario"] == policy]
        values = [
            part[part["mitigation_scenario"] == mitigation][
                "loss_reduction_vs_baseline_pct"
            ].dropna().to_numpy()
            for mitigation in MITIGATION_ORDER
        ]
        box = ax.boxplot(
            values,
            patch_artist=True,
            widths=0.55,
            showfliers=False,
            medianprops={"color": "#111827", "linewidth": 1.5},
            boxprops={"edgecolor": "#334155", "linewidth": 1.1},
            whiskerprops={"color": "#334155", "linewidth": 1.1},
            capprops={"color": "#334155", "linewidth": 1.1},
        )
        for patch in box["boxes"]:
            patch.set_facecolor("#BFD7EA")
            patch.set_alpha(0.75)
        for i, vals in enumerate(values, start=1):
            jitter = rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter,
                vals,
                s=14,
                color="#1F77B4",
                alpha=0.42,
                linewidths=0,
            )
        ax.set_title(
            policy.replace("_policy", "").replace("_", " ").title(),
            loc="left",
            weight="bold",
        )
        ax.set_xticks(range(1, len(MITIGATION_ORDER) + 1))
        ax.set_xticklabels(
            [MITIGATION_LABELS[item] for item in MITIGATION_ORDER],
            fontsize=9,
        )
        ax.set_ylabel("Loss reduction vs baseline, %")
        style(ax)
    fig.suptitle(
        "Fig4 demand-side sensitivity: all mitigation scenarios in 2050",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = OUT_DIR / "fig4_demand_side_sensitivity_all_mitigation_2050.png"
    fig.savefig(out, dpi=240)
    plt.close(fig)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    data = read_results()
    all_path = OUT_DIR / "fig4_demand_side_sensitivity_all.csv"
    data.to_csv(all_path, index=False)
    heatmaps = [plot_2050_heatmap(data, policy) for policy in POLICIES]
    outputs = [
        all_path,
        *write_key_tables(data),
        *heatmaps,
        plot_spread(data),
        plot_all_mitigation_2050(data),
    ]
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
