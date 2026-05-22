from pathlib import Path

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_FILE = (
    ROOT
    / "unified_policy_run"
    / "data"
    / "lithium_loss_scenarios_unified"
    / "lithium_loss_scenarios_summary.csv"
)
OUT_DIR = ROOT / "unified_policy_run" / "figures" / "fig4_mitigation_marginal"

YEARS = [2030, 2040, 2050]
STRATEGY = "Strategy 3"

POLICY_ORDER = [
    "current_policy",
    "reference_policy",
    "strict_policy",
    "critical_route_policy",
]
POLICY_LABELS = {
    "current_policy": "Current",
    "reference_policy": "Ref.",
    "strict_policy": "Strict",
    "critical_route_policy": "Crit.",
}
POLICY_COLORS = {
    "current_policy": "#2C7FB8",
    "reference_policy": "#7FB3D5",
    "strict_policy": "#D7263D",
    "critical_route_policy": "#F4A261",
}

MITIGATION_ORDER = [
    "policy_relaxation",
    "high_recovery_efficiency",
    "high_direct_maturity",
    "capacity_expansion",
    "combined_mitigation",
    "max_lithium",
    "lithium_aware_high_price",
]
MITIGATION_LABELS = {
    "policy_relaxation": "Policy relaxation",
    "high_recovery_efficiency": "High recovery efficiency",
    "high_direct_maturity": "High Direct maturity",
    "capacity_expansion": "Capacity expansion",
    "combined_mitigation": "Combined mitigation",
    "max_lithium": "Max-lithium objective",
    "lithium_aware_high_price": "Li-aware high price",
}


def load_pct():
    df = pd.read_csv(DATA_FILE)
    df = df[(df["strategy"] == STRATEGY) & (df["year"].isin(YEARS))].copy()
    df["available_t"] = df["recovered_lithium_t"] - df["route_access_displaced_lithium_t"]

    rows = []
    for year, sub in df.groupby("year"):
        pivot = sub.pivot_table(
            index="mitigation_scenario",
            columns="policy_scenario",
            values="available_t",
            aggfunc="first",
        )
        base = pivot.loc["baseline"]
        pct = pivot.subtract(base, axis=1).divide(base.replace(0, np.nan), axis=1) * 100.0
        pct = pct.reindex(index=MITIGATION_ORDER, columns=POLICY_ORDER).fillna(0.0)
        long = pct.reset_index().melt(
            id_vars="mitigation_scenario",
            var_name="policy_scenario",
            value_name="marginal_available_li_pct",
        )
        long["year"] = int(year)
        rows.append(long)
    return pd.concat(rows, ignore_index=True)


def main():
    plt.rcParams.update({"font.family": "Arial"})
    data = load_pct()
    min_value = -60.0
    max_value = 60.0
    zero_radius = abs(min_value)
    radial_max = max_value - min_value
    tick_values = np.array([min_value, -30.0, 0.0, 30.0, max_value])
    radial_ticks = zero_radius + tick_values

    n_year = len(YEARS)
    theta_centers = np.linspace(0, 2 * np.pi, n_year, endpoint=False)
    group_width = 2 * np.pi / n_year * 0.72
    bar_width = group_width / len(POLICY_ORDER)
    offsets = (np.arange(len(POLICY_ORDER)) - (len(POLICY_ORDER) - 1) / 2.0) * bar_width

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(15.5, 8.5),
        subplot_kw={"projection": "polar"},
        dpi=300,
    )
    axes = axes.ravel()

    for ax, mitigation in zip(axes, MITIGATION_ORDER):
        sub = data[data["mitigation_scenario"] == mitigation]
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_ylim(0, radial_max)
        ax.set_yticks(radial_ticks)
        ax.set_yticklabels([f"{v:.0f}%" for v in tick_values], fontsize=7)
        ax.set_rlabel_position(86)
        ax.grid(color="#DDE2E6", linewidth=0.65)
        ax.spines["polar"].set_color("#AEB7C2")
        ax.spines["polar"].set_linewidth(0.8)
        theta_line = np.linspace(0, 2 * np.pi, 360)
        ax.plot(theta_line, np.full_like(theta_line, zero_radius), color="#2F3437", linewidth=0.9)

        for pol_idx, policy in enumerate(POLICY_ORDER):
            values = []
            for year in YEARS:
                row = sub[
                    (sub["year"] == year)
                    & (sub["policy_scenario"] == policy)
                ]
                values.append(float(row["marginal_available_li_pct"].iloc[0]) if not row.empty else 0.0)
            raw_values = np.asarray(values)
            values = np.clip(raw_values, min_value, max_value)
            bottoms = np.where(values >= 0, zero_radius, zero_radius + values)
            heights = np.abs(values)
            ax.bar(
                theta_centers + offsets[pol_idx],
                heights,
                width=bar_width * 0.86,
                bottom=bottoms,
                color=POLICY_COLORS[policy],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.94,
            )
            for angle, raw, clipped in zip(theta_centers + offsets[pol_idx], raw_values, values):
                if min_value <= raw <= max_value:
                    continue
                r = zero_radius + clipped
                ha = "left" if np.cos(angle) >= 0 else "right"
                ax.annotate(
                    f"{raw:+.1f}%",
                    xy=(angle, r),
                    xytext=(0, 4 if raw > max_value else -4),
                    textcoords="offset points",
                    ha=ha,
                    va="center",
                    fontsize=6.7,
                    color="#111111",
                    fontweight="bold",
                )

        ax.set_xticks(theta_centers)
        ax.set_xticklabels([str(y) for y in YEARS], fontsize=8)
        ax.tick_params(axis="x", pad=5)
        ax.set_title(MITIGATION_LABELS[mitigation], fontsize=10.2, fontweight="bold", y=1.08)

    axes[-1].axis("off")

    legend_handles = [
        Patch(facecolor=POLICY_COLORS[p], edgecolor="white", label=POLICY_LABELS[p])
        for p in POLICY_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.035),
    )
    fig.suptitle(
        "Relative marginal benefit by mitigation scenario",
        y=0.97,
        fontsize=13.2,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.085,
        "Each circle is one mitigation; angular positions are years, colors are policies, and the dark ring is 0%.",
        ha="center",
        fontsize=8.8,
        color="#374151",
    )
    fig.subplots_adjust(left=0.035, right=0.975, top=0.88, bottom=0.15, wspace=0.24, hspace=0.40)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv = OUT_DIR / "mitigation_marginal_benefit_pct_polar_by_mitigation_2030_2040_2050.csv"
    png = OUT_DIR / "mitigation_marginal_benefit_pct_polar_by_mitigation_2030_2040_2050.png"
    pdf = OUT_DIR / "mitigation_marginal_benefit_pct_polar_by_mitigation_2030_2040_2050.pdf"
    data.to_csv(csv, index=False)
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Wrote {csv}")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
