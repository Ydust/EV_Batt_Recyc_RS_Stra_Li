from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
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
ROUTES_FILE = (
    ROOT
    / "unified_policy_run"
    / "data"
    / "lithium_loss_scenarios_unified"
    / "lithium_loss_scenarios_routes.csv"
)
COUNTRY_FILE = ROOT / "all_countries.csv"
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
    "reference_policy": "Reference",
    "strict_policy": "Strict",
    "critical_route_policy": "Critical-route",
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
    "policy_relaxation": "Policy\nrelax.",
    "high_recovery_efficiency": "Recovery\neff.",
    "high_direct_maturity": "Direct\nmaturity",
    "capacity_expansion": "Capacity\nexp.",
    "combined_mitigation": "Combined",
    "max_lithium": "Max Li",
    "lithium_aware_high_price": "Li-aware\nprice",
}
# One distinct hue per mitigation strategy (used as the dark "available Li"
# shade; the light shade for displaced Li is derived via lighten()).
MITIGATION_COLORS = {
    "policy_relaxation": "#2C7FB8",
    "high_recovery_efficiency": "#2A9D8F",
    "high_direct_maturity": "#8E7CC3",
    "capacity_expansion": "#E9A23B",
    "combined_mitigation": "#D7263D",
    "max_lithium": "#5AAA46",
    "lithium_aware_high_price": "#E76F51",
}


def lighten(color, amount):
    """Blend a colour toward white; amount in [0, 1], higher is lighter."""
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * amount)

# Continents that actually receive recycling routes in this dataset.
CONTINENTS = ["Asia", "Europe", "America", "Pacific"]
RADAR_MITIGATION = "combined_mitigation"


def load_pct_for_year(year):
    df = pd.read_csv(DATA_FILE)
    df = df[(df["strategy"] == STRATEGY) & (df["year"] == year)].copy()
    df["available_t"] = df["recovered_lithium_t"] - df["route_access_displaced_lithium_t"]
    pivot = df.pivot_table(
        index="mitigation_scenario",
        columns="policy_scenario",
        values="available_t",
        aggfunc="first",
    )
    base = pivot.loc["baseline"]
    pct = pivot.subtract(base, axis=1).divide(base.replace(0, np.nan), axis=1) * 100.0
    return pct.reindex(index=MITIGATION_ORDER, columns=POLICY_ORDER).fillna(0.0)


def load_continent_pct():
    """Continent-level recovered-Li marginal benefit (%) of RADAR_MITIGATION vs baseline."""
    continent_map = pd.read_csv(COUNTRY_FILE).set_index("iso3")["continent"].to_dict()
    routes = pd.read_csv(ROUTES_FILE)
    routes = routes[
        (routes["strategy"] == STRATEGY)
        & (~routes["is_unprocessed"].astype(bool))
        & (routes["policy_scenario"].isin(POLICY_ORDER))
    ].copy()
    routes["continent"] = routes["destination_iso3"].map(continent_map).fillna("Other")
    routes = routes[routes["continent"].isin(CONTINENTS)]
    routes["recovered_lithium_t"] = pd.to_numeric(
        routes["recovered_lithium_t"], errors="coerce"
    ).fillna(0.0)
    grouped = routes.groupby(
        ["year", "policy_scenario", "mitigation_scenario", "continent"],
        as_index=False,
    )["recovered_lithium_t"].sum()
    out = {}
    for year in YEARS:
        sub = grouped[grouped["year"] == year]
        base = sub[sub["mitigation_scenario"] == "baseline"].set_index(
            ["policy_scenario", "continent"]
        )["recovered_lithium_t"]
        comb = sub[sub["mitigation_scenario"] == RADAR_MITIGATION].set_index(
            ["policy_scenario", "continent"]
        )["recovered_lithium_t"]
        idx = pd.MultiIndex.from_product(
            [POLICY_ORDER, CONTINENTS], names=["policy_scenario", "continent"]
        )
        base = base.reindex(idx)
        comb = comb.reindex(idx)
        pct = (comb - base).divide(base.replace(0, np.nan)) * 100.0
        out[year] = pct.unstack().reindex(index=POLICY_ORDER, columns=CONTINENTS).fillna(0.0)
    return out


def load_breakdown_by_mitigation():
    """Available Li vs route-access-displaced Li per policy/mitigation/year."""
    df = pd.read_csv(DATA_FILE)
    df = df[
        (df["strategy"] == STRATEGY)
        & (df["mitigation_scenario"].isin(MITIGATION_ORDER))
    ].copy()
    df["available_t"] = df["recovered_lithium_t"] - df["route_access_displaced_lithium_t"]
    out = {}
    for year in YEARS:
        sub = df[df["year"] == year]
        avail = sub.pivot_table(
            index="mitigation_scenario",
            columns="policy_scenario",
            values="available_t",
            aggfunc="first",
        ).reindex(index=MITIGATION_ORDER, columns=POLICY_ORDER).fillna(0.0)
        disp = sub.pivot_table(
            index="mitigation_scenario",
            columns="policy_scenario",
            values="route_access_displaced_lithium_t",
            aggfunc="first",
        ).reindex(index=MITIGATION_ORDER, columns=POLICY_ORDER).fillna(0.0)
        out[year] = {"available_t": avail, "route_access_displaced_lithium_t": disp}
    return out


def draw_polar(ax, year, data, panel_label):
    min_value, max_value = -60.0, 60.0
    zero_radius = abs(min_value)
    radial_max = max_value - min_value
    tick_values = np.array([min_value, -30.0, 0.0, 30.0, max_value])
    radial_ticks = zero_radius + tick_values

    n_mit = len(MITIGATION_ORDER)
    n_pol = len(POLICY_ORDER)
    sector_width = 2 * np.pi / n_mit
    theta_centers = np.linspace(0, 2 * np.pi, n_mit, endpoint=False)
    sector_edges = theta_centers - sector_width / 2.0
    group_width = sector_width * 0.82
    bar_width = group_width / n_pol
    offsets = (np.arange(n_pol) - (n_pol - 1) / 2.0) * bar_width

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, radial_max)
    ax.set_yticks(radial_ticks)
    ax.set_yticklabels([f"{tick:.0f}%" for tick in tick_values], fontsize=9.5)
    ax.set_rlabel_position(88)
    ax.grid(axis="y", color="#DDE2E6", linewidth=0.7)
    ax.grid(axis="x", visible=False)
    ax.spines["polar"].set_color("#AEB7C2")
    ax.spines["polar"].set_linewidth(0.8)
    theta_line = np.linspace(0, 2 * np.pi, 360)
    ax.plot(theta_line, np.full_like(theta_line, zero_radius), color="#2F3437", linewidth=1.0)
    for edge in sector_edges:
        ax.plot([edge, edge], [0, radial_max], color="#E6EBF0", linewidth=0.8, zorder=0)
    for idx, center in enumerate(theta_centers):
        if idx % 2 == 0:
            ax.bar(
                center,
                radial_max,
                width=sector_width * 0.96,
                bottom=0,
                color="#F7F9FB",
                edgecolor="none",
                alpha=0.55,
                zorder=-2,
            )

    for pol_idx, policy in enumerate(POLICY_ORDER):
        raw_values = data[policy].to_numpy()
        values = np.clip(raw_values, min_value, max_value)
        theta = theta_centers + offsets[pol_idx]
        bottoms = np.where(values >= 0, zero_radius, zero_radius + values)
        heights = np.abs(values)
        ax.bar(
            theta,
            heights,
            width=bar_width * 0.88,
            bottom=bottoms,
            color=POLICY_COLORS[policy],
            edgecolor="white",
            linewidth=0.55,
            alpha=0.94,
        )
        for angle, raw, clipped in zip(theta, raw_values, values):
            if min_value <= raw <= max_value:
                continue
            r = zero_radius + clipped
            ha = "left" if np.cos(angle) >= 0 else "right"
            text_shift = (pol_idx - (n_pol - 1) / 2.0) * 12.0
            ax.annotate(
                f"{raw:+.1f}%",
                xy=(angle, r),
                xytext=(text_shift, 10 if raw > max_value else -10),
                textcoords="offset points",
                ha=ha,
                va="center",
                fontsize=8.6,
                color="#111111",
                fontweight="bold",
            )

    ax.set_xticks([])
    for center, mitigation in zip(theta_centers, MITIGATION_ORDER):
        ax.text(
            center,
            radial_max * 1.08,
            MITIGATION_LABELS[mitigation],
            ha="center",
            va="center",
            fontsize=8.8,
            clip_on=False,
        )
    ax.set_title(f"{year}", y=1.10, fontsize=16.0, fontweight="bold")
    ax.text(
        -0.08,
        1.10,
        panel_label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
    )


def draw_radar(ax, year, pct_df, limit, panel_label):
    """Continent-axis radar: one line per policy, value = combined-mitigation % vs baseline."""
    angles = np.linspace(0, 2 * np.pi, len(CONTINENTS), endpoint=False).tolist()
    angles += angles[:1]
    for policy in POLICY_ORDER:
        raw = [float(pct_df.loc[policy, c]) for c in CONTINENTS]
        clipped = [max(-limit, min(limit, v)) for v in raw]
        scaled = [0.5 + 0.5 * v / limit for v in clipped]
        scaled += scaled[:1]
        ax.plot(
            angles,
            scaled,
            color=POLICY_COLORS[policy],
            linewidth=2.0,
            alpha=0.92,
            marker="o",
            markersize=3.5,
        )
    ax.set_facecolor("none")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(CONTINENTS, fontsize=9.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.grid(False)
    ring_theta = np.linspace(0, 2 * np.pi, 200)
    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(ring_theta, [r] * len(ring_theta), color="0.80", linewidth=0.7, zorder=1)
    ax.plot(ring_theta, [0.5] * len(ring_theta), color="#2F3437", linewidth=1.0, zorder=2)
    ax.scatter([0], [0], color="0.78", s=8, zorder=1)
    label_theta = np.pi / 4  # gap between Asia and Pacific
    for r_scaled, pct_value in [(0.0, -limit), (0.5, 0.0), (1.0, limit)]:
        sign = "+" if pct_value > 0 else ("−" if pct_value < 0 else "")
        ax.text(
            label_theta,
            r_scaled,
            f"{sign}{abs(int(pct_value))}%",
            fontsize=7.5,
            color="0.40",
            ha="center",
            va="center",
            zorder=5,
            bbox=dict(
                boxstyle="round,pad=0.18",
                facecolor="white",
                edgecolor="0.75",
                linewidth=0.4,
                alpha=0.95,
            ),
        )
    for spine in ax.spines.values():
        spine.set_color("0.78")
        spine.set_linewidth(0.8)
    ax.tick_params(axis="x", pad=4)
    ax.set_title(f"{year}", y=1.12, fontsize=13.5, fontweight="bold")
    ax.text(
        -0.08,
        1.16,
        panel_label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
    )


def draw_breakdown(ax, year, breakdown, panel_label):
    """Grouped stacked bars: per policy scenario, one bar per mitigation strategy;
    dark shade = available Li, light shade = route-access-displaced Li."""
    avail = breakdown["available_t"]
    disp = breakdown["route_access_displaced_lithium_t"]
    n_pol = len(POLICY_ORDER)
    n_mit = len(MITIGATION_ORDER)
    x = np.arange(n_pol)
    group_width = 0.86
    bar_width = group_width / n_mit
    offsets = (np.arange(n_mit) - (n_mit - 1) / 2.0) * bar_width

    tops = []
    for m_idx, mitigation in enumerate(MITIGATION_ORDER):
        base = MITIGATION_COLORS[mitigation]
        a = avail.loc[mitigation, POLICY_ORDER].to_numpy() / 1000.0
        d = disp.loc[mitigation, POLICY_ORDER].to_numpy() / 1000.0
        xpos = x + offsets[m_idx]
        ax.bar(
            xpos,
            a,
            width=bar_width * 0.9,
            color=base,
            edgecolor="white",
            linewidth=0.4,
        )
        ax.bar(
            xpos,
            d,
            width=bar_width * 0.9,
            bottom=a,
            color=lighten(base, 0.62),
            edgecolor="white",
            linewidth=0.4,
        )
        tops.append(a + d)
    top_max = float(np.max(tops)) if tops else 1.0
    headroom = top_max * 0.10 if top_max > 0 else 1.0
    ax.set_ylim(0, top_max + headroom)
    ax.set_xticks(x)
    ax.set_xticklabels([POLICY_LABELS[p] for p in POLICY_ORDER], fontsize=9.0)
    ax.set_ylabel("Supply-chain Li (kt)", fontsize=10)
    ax.grid(axis="y", color="#E3E8EC", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=9, direction="in")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.set_title(f"{year}", y=1.04, fontsize=13.5, fontweight="bold")
    ax.text(
        -0.12,
        1.12,
        panel_label,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
    )


def main():
    plt.rcParams.update({"font.family": "Arial"})
    pct_by_year = {year: load_pct_for_year(year) for year in YEARS}
    continent_pct = load_continent_pct()
    breakdown = load_breakdown_by_mitigation()
    radar_limit = 60.0

    fig = plt.figure(figsize=(18.8, 21.6), dpi=300)
    grid = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.0, 0.92, 0.74],
        hspace=0.72,
        wspace=0.40,
        left=0.058,
        right=0.93,
        top=0.92,
        bottom=0.155,
    )
    polar_axes = [fig.add_subplot(grid[0, i], projection="polar") for i in range(3)]
    radar_axes = [fig.add_subplot(grid[1, i], projection="polar") for i in range(3)]
    bar_axes = [fig.add_subplot(grid[2, i]) for i in range(3)]

    polar_labels = ["a", "b", "c"]
    radar_labels = ["d", "e", "f"]
    bar_labels = ["g", "h", "i"]

    for ax, year, label in zip(polar_axes, YEARS, polar_labels):
        draw_polar(ax, year, pct_by_year[year], label)
    for ax, year, label in zip(radar_axes, YEARS, radar_labels):
        draw_radar(ax, year, continent_pct[year], radar_limit, label)
    for ax, year, label in zip(bar_axes, YEARS, bar_labels):
        draw_breakdown(ax, year, breakdown[year], label)

    policy_handles = [
        Patch(facecolor=POLICY_COLORS[p], edgecolor="white", label=POLICY_LABELS[p])
        for p in POLICY_ORDER
    ]
    leg_policy = fig.legend(
        handles=policy_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=12,
        title="Policy scenario (rows a-c bars, rows d-f lines)",
        title_fontsize=11,
        bbox_to_anchor=(0.5, 0.058),
    )
    fig.add_artist(leg_policy)
    mitigation_handles = [
        Patch(
            facecolor=MITIGATION_COLORS[m],
            edgecolor="white",
            label=MITIGATION_LABELS[m].replace("\n", " "),
        )
        for m in MITIGATION_ORDER
    ]
    fig.legend(
        handles=mitigation_handles,
        loc="upper center",
        ncol=7,
        frameon=False,
        fontsize=10,
        title="Mitigation strategy (rows g-i bars; dark = available Li, light = Li displaced by route-access policy)",
        title_fontsize=10.5,
        bbox_to_anchor=(0.5, 0.026),
    )

    fig.suptitle(
        "Relative marginal benefit on supply-chain-available Li",
        y=0.965,
        fontsize=18.0,
        fontweight="bold",
    )

    def row_box(axes_list):
        boxes = [ax.get_position() for ax in axes_list]
        return (
            min(b.y0 for b in boxes),
            max(b.y1 for b in boxes),
        )

    polar_y0, polar_y1 = row_box(polar_axes)
    radar_y0, radar_y1 = row_box(radar_axes)
    bar_y0, bar_y1 = row_box(bar_axes)

    # Row descriptors on the left margin, centred on each row.
    for y_mid, label in [
        ((polar_y0 + polar_y1) / 2, "Mitigation × policy"),
        ((radar_y0 + radar_y1) / 2, "Continent breakdown"),
        ((bar_y0 + bar_y1) / 2, "Mitigation breakdown"),
    ]:
        fig.text(
            0.014,
            y_mid,
            label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#374151",
        )

    # Per-row captions placed in the gap just below each row.
    fig.text(
        0.5,
        polar_y0 - 0.030,
        "Rows a-c: each mitigation occupies one circular sector; radial bars show % change vs baseline available Li, the dark ring is 0%.",
        ha="center",
        fontsize=10.5,
        color="#374151",
    )
    fig.text(
        0.5,
        radar_y0 - 0.026,
        "Rows d-f: continent-level recovered-Li marginal benefit (%) of the Combined mitigation vs baseline; the dark ring is 0%.",
        ha="center",
        fontsize=10.5,
        color="#374151",
    )
    fig.text(
        0.5,
        bar_y0 - 0.052,
        "Rows g-i: for each policy scenario the seven coloured bars are the seven mitigation strategies; within each bar the dark\n"
        "segment is available Li (recovered − displaced) and the light segment is Li displaced by route-access policy.",
        ha="center",
        va="top",
        fontsize=10.5,
        color="#374151",
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long_rows = []
    for year, d in pct_by_year.items():
        out = d.reset_index().melt(
            id_vars="mitigation_scenario",
            var_name="policy_scenario",
            value_name="marginal_available_li_pct",
        )
        out["year"] = year
        long_rows.append(out)
    csv = OUT_DIR / "mitigation_marginal_benefit_pct_polar_2030_2040_2050.csv"
    pd.concat(long_rows, ignore_index=True).to_csv(csv, index=False)

    continent_rows = []
    for year, d in continent_pct.items():
        out = d.reset_index().melt(
            id_vars="policy_scenario",
            var_name="continent",
            value_name="combined_marginal_recovered_li_pct",
        )
        out["year"] = year
        continent_rows.append(out)
    continent_csv = OUT_DIR / "mitigation_marginal_benefit_continent_radar_2030_2040_2050.csv"
    pd.concat(continent_rows, ignore_index=True).to_csv(continent_csv, index=False)

    breakdown_rows = []
    for year, d in breakdown.items():
        avail = d["available_t"].reset_index().melt(
            id_vars="mitigation_scenario",
            var_name="policy_scenario",
            value_name="available_t",
        )
        disp = d["route_access_displaced_lithium_t"].reset_index().melt(
            id_vars="mitigation_scenario",
            var_name="policy_scenario",
            value_name="route_access_displaced_lithium_t",
        )
        merged = avail.merge(disp, on=["mitigation_scenario", "policy_scenario"])
        merged["year"] = year
        breakdown_rows.append(merged)
    breakdown_csv = OUT_DIR / "mitigation_marginal_benefit_baseline_breakdown_2030_2040_2050.csv"
    pd.concat(breakdown_rows, ignore_index=True).to_csv(breakdown_csv, index=False)

    png = OUT_DIR / "mitigation_marginal_benefit_pct_polar_2030_2040_2050.png"
    pdf = OUT_DIR / "mitigation_marginal_benefit_pct_polar_2030_2040_2050.pdf"
    fig.savefig(png, dpi=240)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Wrote {csv}")
    print(f"Wrote {continent_csv}")
    print(f"Wrote {breakdown_csv}")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
