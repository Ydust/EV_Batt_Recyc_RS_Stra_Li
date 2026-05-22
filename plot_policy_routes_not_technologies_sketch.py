from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "Figure_data" / "joint_policy_technology" / "sketch"
ELASTICITY_DIR = (
    ROOT
    / "Figure_data"
    / "joint_policy_technology"
    / "policy_objective_technology_elasticity"
)
DIRECT_DIR = ROOT / "Figure_data" / "joint_policy_technology" / "direct_entry_cost"
MECH_DIR = ROOT / "Figure_data" / "joint_policy_technology" / "policy_technology_mechanisms"

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
REGION_ORDER = ["China", "United States", "European Union"]
REGION_LABELS = {"United States": "US", "European Union": "EU", "China": "China"}


def read_elasticity_mix():
    frames = []
    for source in ["", "aggregate_us_eu"]:
        path = (
            ELASTICITY_DIR / source / "policy_objective_technology_elasticity_mix.csv"
            if source
            else ELASTICITY_DIR / "policy_objective_technology_elasticity_mix.csv"
        )
        if path.exists():
            frames.append(pd.read_csv(path))
    data = pd.concat(frames, ignore_index=True)
    data["year"] = pd.to_numeric(data["year"], errors="coerce").astype(int)
    data["target_share_of_max"] = pd.to_numeric(data["target_share_of_max"], errors="coerce")
    data["technology_share_pct"] = pd.to_numeric(data["technology_share_pct"], errors="coerce")
    return data


def panel_a(ax, mix):
    data = mix[
        (mix["year"] == 2050)
        & (mix["scope"] == "target_region")
        & (mix["technology"] == "Direct")
        & (mix["target_region"].isin(REGION_ORDER))
        & (mix["policy_scenario"].isin(POLICY_ORDER))
    ].copy()
    colors = {"China": "#1f77b4", "United States": "#2ca02c", "European Union": "#d62728"}
    linestyles = {
        "reference_policy": "-",
        "current_policy": "--",
        "strict_policy": ":",
        "critical_route_policy": "-.",
    }
    for region in REGION_ORDER:
        for policy in POLICY_ORDER:
            subset = data[
                (data["target_region"] == region) & (data["policy_scenario"] == policy)
            ].sort_values("target_share_of_max")
            if subset.empty:
                continue
            ax.plot(
                subset["target_share_of_max"] * 100,
                subset["technology_share_pct"],
                color=colors[region],
                linestyle=linestyles[policy],
                linewidth=1.7,
                alpha=0.9,
            )
    ax.axvline(99, color="0.25", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.text(99.3, 8, "Direct enters\nnear 99%", fontsize=8, color="0.25")
    ax.set_title("A. Direct share responds to target stringency")
    ax.set_xlabel("Regional Li-access target (% of max)")
    ax.set_ylabel("Direct share in target region (%)")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-3, 105)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    region_handles = [
        plt.Line2D([0], [0], color=colors[r], lw=2, label=REGION_LABELS[r]) for r in REGION_ORDER
    ]
    policy_handles = [
        plt.Line2D([0], [0], color="0.25", lw=1.7, linestyle=linestyles[p], label=POLICY_LABELS[p])
        for p in POLICY_ORDER
    ]
    leg1 = ax.legend(handles=region_handles, loc="upper left", frameon=False, fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=policy_handles, loc="lower right", frameon=False, fontsize=8)


def panel_b(ax):
    thresholds = pd.read_csv(DIRECT_DIR / "direct_entry_thresholds.csv")
    thresholds["year"] = pd.to_numeric(thresholds["year"], errors="coerce").astype(int)
    thresholds["direct_share_threshold_pct"] = pd.to_numeric(
        thresholds["direct_share_threshold_pct"], errors="coerce"
    )
    thresholds["direct_entry_target_share_pct"] = pd.to_numeric(
        thresholds["direct_entry_target_share_pct"], errors="coerce"
    )
    data = thresholds[
        (thresholds["scope"] == "target_region")
        & (thresholds["direct_share_threshold_pct"] == 5.0)
        & (thresholds["target_region"].isin(REGION_ORDER))
        & (thresholds["policy_scenario"].isin(POLICY_ORDER))
    ].copy()
    rows = []
    labels = []
    for region in REGION_ORDER:
        for policy in POLICY_ORDER:
            row = []
            for year in [2030, 2040, 2050]:
                match = data[
                    (data["target_region"] == region)
                    & (data["policy_scenario"] == policy)
                    & (data["year"] == year)
                ]
                row.append(float(match["direct_entry_target_share_pct"].iloc[0]) if not match.empty else np.nan)
            rows.append(row)
            labels.append(f"{REGION_LABELS[region]} - {POLICY_LABELS[policy]}")
    matrix = np.array(rows)
    image = ax.imshow(matrix, vmin=95, vmax=100, cmap="YlGnBu", aspect="auto")
    ax.set_title("B. Direct entry threshold is policy-invariant")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["2030", "2040", "2050"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    for y in range(matrix.shape[0]):
        for x in range(matrix.shape[1]):
            value = matrix[y, x]
            if np.isfinite(value):
                ax.text(x, y, f"{value:.0f}%", ha="center", va="center", fontsize=7, color="black")
    cbar = plt.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Target share at Direct >5%")


def panel_c(ax):
    shift_path = MECH_DIR / "destination_shift_vs_current.csv"
    value_col = "destination_recovered_li_delta_vs_current_t"
    if not shift_path.exists():
        shift_path = MECH_DIR / "destination_shift_vs_reference.csv"
        value_col = "destination_recovered_li_delta_vs_reference_t"
    shift = pd.read_csv(shift_path)
    shift["year"] = pd.to_numeric(shift["year"], errors="coerce").astype(int)
    shift[value_col] = pd.to_numeric(
        shift[value_col], errors="coerce"
    )
    data = shift[
        (shift["year"] == 2050)
        & (shift["policy_scenario"].isin(["reference_policy", "strict_policy", "critical_route_policy"]))
    ].copy()
    top_destinations = (
        data.groupby("destination_iso3")[value_col]
        .apply(lambda x: x.abs().max())
        .sort_values(ascending=False)
        .head(8)
        .index.tolist()
    )
    plot_data = data[data["destination_iso3"].isin(top_destinations)].copy()
    pivot = plot_data.pivot_table(
        index="destination_iso3",
        columns="policy_scenario",
        values=value_col,
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(top_destinations)
    x = np.arange(len(pivot.index))
    width = 0.25
    ax.bar(
        x - width,
        pivot.get("reference_policy", pd.Series(0, index=pivot.index)) / 1000,
        width=width,
        color="#4E79A7",
        label="Reference",
    )
    ax.bar(
        x,
        pivot.get("strict_policy", pd.Series(0, index=pivot.index)) / 1000,
        width=width,
        color="#E15759",
        label="Strict",
    )
    ax.bar(
        x + width,
        pivot.get("critical_route_policy", pd.Series(0, index=pivot.index)) / 1000,
        width=width,
        color="#59A14F",
        label="Critical-route",
    )
    ax.axhline(0, color="0.25", linewidth=0.8)
    ax.set_title("C. Policy mainly reallocates destination flows")
    ax.set_xlabel("Destination")
    ax.set_ylabel("Recovered Li delta vs Current (kt)")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=0)
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.legend(frameon=False, fontsize=8)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mix = read_elasticity_mix()
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )
    fig = plt.figure(figsize=(14, 7.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])
    panel_a(ax_a, mix)
    panel_b(ax_b)
    panel_c(ax_c)
    fig.suptitle(
        "Policy constraints reshape recycling geography more than technology selection",
        fontsize=13,
        fontweight="bold",
        y=0.985,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    png = OUT_DIR / "policy_routes_not_technologies_sketch.png"
    pdf = OUT_DIR / "policy_routes_not_technologies_sketch.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
