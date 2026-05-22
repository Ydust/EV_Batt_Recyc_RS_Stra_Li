from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RUN_DIR = ROOT / "unified_policy_run"
SENS_DIR = RUN_DIR / "sensitivity_runs"
OUT_DIR = RUN_DIR / "figures" / "sensitivity_by_category"

FIG3_BASE = RUN_DIR / "data" / "fig3_pyrohydro_sensitivity_unified"
FIG4_BASE = RUN_DIR / "data" / "lithium_loss_scenarios_unified"

S_CASES = {
    "pyrohydro_sensitivity_conservative_unified": "S1 conservative",
    "pyrohydro_sensitivity_s2_unified": "S2 low-mid",
    "pyrohydro_sensitivity_medium_unified": "S3 medium",
    "pyrohydro_sensitivity_s4_unified": "S4 mid-high",
    "pyrohydro_sensitivity_s5_unified": "S5 strong",
}

CONFIGS = [
    {
        "config": "baseline",
        "category": "Baseline",
        "label": "baseline",
        "level": 0.0,
        "fig3_dir": FIG3_BASE,
        "fig4_dir": FIG4_BASE,
    },
    {
        "config": "direct_mult_1.0",
        "category": "Direct cost multiplier",
        "label": "x1.0",
        "level": 1.0,
    },
    {
        "config": "direct_mult_1.5",
        "category": "Direct cost multiplier",
        "label": "x1.5",
        "level": 1.5,
    },
    {
        "config": "avail_thr_0.4",
        "category": "Technology availability threshold",
        "label": "0.4",
        "level": 0.4,
    },
    {
        "config": "avail_thr_0.8",
        "category": "Technology availability threshold",
        "label": "0.8",
        "level": 0.8,
    },
    {
        "config": "penalty_150",
        "category": "China policy penalty",
        "label": "$150/t",
        "level": 150.0,
    },
    {
        "config": "penalty_600",
        "category": "China policy penalty",
        "label": "$600/t",
        "level": 600.0,
    },
    {
        "config": "delay_5",
        "category": "Delay cost",
        "label": "$5/t-day",
        "level": 5.0,
    },
    {
        "config": "delay_10",
        "category": "Delay cost",
        "label": "$10/t-day",
        "level": 10.0,
    },
    {
        "config": "li_price_3",
        "category": "Lithium price multiplier",
        "label": "x3",
        "level": 3.0,
        "fig3": False,
    },
    {
        "config": "li_price_5",
        "category": "Lithium price multiplier",
        "label": "x5",
        "level": 5.0,
        "fig3": False,
    },
]

POLICY_ORDER = [
    "reference_policy",
    "current_policy",
    "strict_policy",
    "critical_route_policy",
    "open_policy",
]
POLICY_COLORS = {
    "reference_policy": "#4C78A8",
    "current_policy": "#F58518",
    "strict_policy": "#54A24B",
    "critical_route_policy": "#B279A2",
    "open_policy": "#72B7B2",
}


def config_fig3_dir(meta):
    if meta.get("fig3") is False:
        return None
    if "fig3_dir" in meta:
        return meta["fig3_dir"]
    return SENS_DIR / meta["config"] / "fig3_pyrohydro_sensitivity_unified"


def config_fig4_dir(meta):
    if "fig4_dir" in meta:
        return meta["fig4_dir"]
    return SENS_DIR / meta["config"] / "lithium_loss_scenarios_unified"


def read_fig3():
    frames = []
    for meta in CONFIGS:
        base = config_fig3_dir(meta)
        if base is None or not base.exists():
            continue
        for folder, case_label in S_CASES.items():
            path = base / folder / "dynamic_scale_summary.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            df["config"] = meta["config"]
            df["category"] = meta["category"]
            df["config_label"] = meta["label"]
            df["config_level"] = meta["level"]
            df["s_case"] = case_label
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def read_fig4():
    frames = []
    for meta in CONFIGS:
        base = config_fig4_dir(meta)
        path = base / "lithium_loss_scenarios_summary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["config"] = meta["config"]
        df["category"] = meta["category"]
        df["config_label"] = meta["label"]
        df["config_level"] = meta["level"]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)


def plot_fig3_delta(fig3):
    medium = fig3[fig3["s_case"] == "S3 medium"].copy()
    agg = (
        medium.groupby(["config", "category", "config_label", "config_level", "policy_scenario"], as_index=False)
        ["recovered_lithium_t"]
        .sum()
    )
    base = agg[agg["config"] == "baseline"][
        ["policy_scenario", "recovered_lithium_t"]
    ].rename(columns={"recovered_lithium_t": "baseline_recovered_lithium_t"})
    agg = agg.merge(base, on="policy_scenario", how="left")
    agg["delta_kt_li"] = (
        agg["recovered_lithium_t"] - agg["baseline_recovered_lithium_t"]
    ) / 1_000
    agg = agg[agg["config"] != "baseline"]

    categories = [
        "Direct cost multiplier",
        "Technology availability threshold",
        "China policy penalty",
        "Delay cost",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=False)
    axes = axes.ravel()
    for ax, category in zip(axes, categories):
        sub = agg[agg["category"] == category].sort_values("config_level")
        labels = sub[["config_label", "config_level"]].drop_duplicates().sort_values("config_level")
        x = np.arange(len(labels))
        width = 0.15
        for i, policy in enumerate(POLICY_ORDER):
            psub = sub[sub["policy_scenario"] == policy].set_index("config_label")
            vals = [psub.loc[label, "delta_kt_li"] if label in psub.index else np.nan for label in labels["config_label"]]
            ax.bar(
                x + (i - 2) * width,
                vals,
                width=width,
                label=policy.replace("_policy", "").replace("_", " "),
                color=POLICY_COLORS.get(policy, "#666666"),
            )
        ax.axhline(0, color="#2F3437", linewidth=0.9)
        ax.set_title(category, loc="left", fontsize=11, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels["config_label"], fontsize=9)
        ax.set_ylabel("Delta vs baseline, kt Li")
        style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("Fig3 sensitivity: cumulative recovered lithium change, S3 medium", fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out = OUT_DIR / "fig3_s3_recovered_li_delta_by_category.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_fig3_s_range(fig3):
    agg = (
        fig3.groupby(["config", "category", "config_label", "config_level", "s_case", "policy_scenario"], as_index=False)
        ["recovered_lithium_t"]
        .sum()
    )
    base = agg[agg["config"] == "baseline"][
        ["s_case", "policy_scenario", "recovered_lithium_t"]
    ].rename(columns={"recovered_lithium_t": "baseline_recovered_lithium_t"})
    agg = agg.merge(base, on=["s_case", "policy_scenario"], how="left")
    agg["delta_kt_li"] = (
        agg["recovered_lithium_t"] - agg["baseline_recovered_lithium_t"]
    ) / 1_000
    agg = agg[(agg["config"] != "baseline") & (agg["policy_scenario"] != "open_policy")]
    ranges = (
        agg.groupby(["config", "category", "config_label", "config_level", "policy_scenario"], as_index=False)
        .agg(min_delta=("delta_kt_li", "min"), max_delta=("delta_kt_li", "max"))
    )
    medium = agg[agg["s_case"] == "S3 medium"][
        ["config", "policy_scenario", "delta_kt_li"]
    ].rename(columns={"delta_kt_li": "medium_delta"})
    ranges = ranges.merge(medium, on=["config", "policy_scenario"], how="left")

    categories = [
        "Direct cost multiplier",
        "Technology availability threshold",
        "China policy penalty",
        "Delay cost",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=False)
    axes = axes.ravel()
    for ax, category in zip(axes, categories):
        sub = ranges[ranges["category"] == category].sort_values(["config_level", "policy_scenario"])
        configs = sub[["config", "config_label", "config_level"]].drop_duplicates().sort_values("config_level")
        x_base = np.arange(len(configs))
        offsets = {
            "reference_policy": -0.24,
            "current_policy": -0.08,
            "strict_policy": 0.08,
            "critical_route_policy": 0.24,
        }
        for policy, offset in offsets.items():
            psub = sub[sub["policy_scenario"] == policy].set_index("config")
            xs, y, yerr_low, yerr_high = [], [], [], []
            for _, row in configs.iterrows():
                if row["config"] not in psub.index:
                    continue
                rec = psub.loc[row["config"]]
                xs.append(x_base[list(configs["config"]).index(row["config"])] + offset)
                y.append(rec["medium_delta"])
                yerr_low.append(rec["medium_delta"] - rec["min_delta"])
                yerr_high.append(rec["max_delta"] - rec["medium_delta"])
            ax.errorbar(
                xs,
                y,
                yerr=[yerr_low, yerr_high],
                fmt="o",
                capsize=3,
                linewidth=1.4,
                markersize=4,
                label=policy.replace("_policy", "").replace("_", " "),
                color=POLICY_COLORS.get(policy, "#666666"),
            )
        ax.axhline(0, color="#2F3437", linewidth=0.9)
        ax.set_xticks(x_base)
        ax.set_xticklabels(configs["config_label"], fontsize=9)
        ax.set_title(category, loc="left", fontsize=11, weight="bold")
        ax.set_ylabel("Delta vs baseline, kt Li\n(point=S3, bar=S1-S5)")
        style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Fig3 sensitivity: S1-S5 range of cumulative recovered lithium change", fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    out = OUT_DIR / "fig3_s1_s5_recovered_li_delta_range_by_category.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def plot_fig4_heatmap(fig4):
    df = fig4[fig4["mitigation_scenario"] != "baseline"].copy()
    df["loss_reduction_vs_baseline_pct"] = pd.to_numeric(
        df["loss_reduction_vs_baseline_pct"], errors="coerce"
    )
    agg = (
        df.groupby(["category", "config", "config_label", "config_level", "mitigation_scenario"], as_index=False)
        ["loss_reduction_vs_baseline_pct"]
        .mean()
    )
    categories = [
        "Direct cost multiplier",
        "Technology availability threshold",
        "China policy penalty",
        "Delay cost",
        "Lithium price multiplier",
    ]
    mitigations = [
        "capacity_expansion",
        "high_direct_maturity",
        "high_recovery_efficiency",
        "lithium_aware_high_price",
        "policy_relaxation",
        "combined_mitigation",
        "max_lithium",
    ]
    fig, axes = plt.subplots(len(categories), 1, figsize=(12, 13))
    vmax = np.nanpercentile(np.abs(agg["loss_reduction_vs_baseline_pct"]), 98)
    vmax = max(vmax, 1)
    for ax, category in zip(axes, categories):
        sub = agg[agg["category"] == category].sort_values("config_level")
        configs = sub[["config", "config_label", "config_level"]].drop_duplicates().sort_values("config_level")
        matrix = []
        for _, cfg in configs.iterrows():
            row = []
            csub = sub[sub["config"] == cfg["config"]].set_index("mitigation_scenario")
            for mitigation in mitigations:
                row.append(
                    csub.loc[mitigation, "loss_reduction_vs_baseline_pct"]
                    if mitigation in csub.index
                    else np.nan
                )
            matrix.append(row)
        arr = np.array(matrix, dtype=float)
        im = ax.imshow(arr, aspect="auto", cmap="YlGnBu", vmin=0, vmax=vmax)
        ax.set_title(category, loc="left", fontsize=11, weight="bold")
        ax.set_yticks(np.arange(len(configs)))
        ax.set_yticklabels(configs["config_label"], fontsize=9)
        ax.set_xticks(np.arange(len(mitigations)))
        ax.set_xticklabels([m.replace("_", " ") for m in mitigations], rotation=25, ha="right", fontsize=8)
        for y in range(arr.shape[0]):
            for x in range(arr.shape[1]):
                if np.isfinite(arr[y, x]):
                    color = "white" if arr[y, x] > vmax * 0.55 else "#172026"
                    ax.text(x, y, f"{arr[y, x]:.1f}", ha="center", va="center", fontsize=7, color=color)
    cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.015)
    cbar.set_label("Mean loss reduction vs scenario baseline, %")
    fig.suptitle("Fig4 sensitivity: mitigation effectiveness by parameter category", fontsize=14, weight="bold")
    fig.tight_layout(rect=(0, 0, 0.96, 0.97))
    out = OUT_DIR / "fig4_mitigation_loss_reduction_heatmap_by_category.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def write_summary_tables(fig3, fig4):
    fig3_agg = (
        fig3.groupby(["config", "category", "config_label", "s_case", "policy_scenario"], as_index=False)
        ["recovered_lithium_t"]
        .sum()
    )
    fig3_base = fig3_agg[fig3_agg["config"] == "baseline"][
        ["s_case", "policy_scenario", "recovered_lithium_t"]
    ].rename(columns={"recovered_lithium_t": "baseline_recovered_lithium_t"})
    fig3_agg = fig3_agg.merge(fig3_base, on=["s_case", "policy_scenario"], how="left")
    fig3_agg["delta_recovered_lithium_t"] = (
        fig3_agg["recovered_lithium_t"] - fig3_agg["baseline_recovered_lithium_t"]
    )
    fig3_path = OUT_DIR / "fig3_recovered_lithium_sensitivity_summary.csv"
    fig3_agg.to_csv(fig3_path, index=False)

    fig4_agg = (
        fig4.groupby(["config", "category", "config_label", "mitigation_scenario", "policy_scenario", "year"], as_index=False)
        .agg(
            total_lithium_loss_t=("total_lithium_loss_t", "sum"),
            loss_reduction_vs_baseline_t=("loss_reduction_vs_baseline_t", "sum"),
            loss_reduction_vs_baseline_pct=("loss_reduction_vs_baseline_pct", "mean"),
        )
    )
    fig4_path = OUT_DIR / "fig4_lithium_loss_sensitivity_summary.csv"
    fig4_agg.to_csv(fig4_path, index=False)
    return fig3_path, fig4_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig3 = read_fig3()
    fig4 = read_fig4()
    outputs = []
    if not fig3.empty:
        outputs.append(plot_fig3_delta(fig3))
        outputs.append(plot_fig3_s_range(fig3))
    if not fig4.empty:
        outputs.append(plot_fig4_heatmap(fig4))
    outputs.extend(write_summary_tables(fig3, fig4))
    for out in outputs:
        print(out)


if __name__ == "__main__":
    main()
