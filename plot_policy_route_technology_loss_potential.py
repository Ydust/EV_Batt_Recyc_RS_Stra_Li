from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "Figure_data" / "joint_policy_technology"
ROUTES_FILE = OUTPUT_DIR / "lithium_loss_scenarios" / "lithium_loss_scenarios_routes.csv"
SUMMARY_FILE = OUTPUT_DIR / "lithium_loss_scenarios" / "lithium_loss_scenarios_summary.csv"
COUNTRY_FILE = ROOT / "all_countries.csv"

YEAR = 2050
POLICIES = [
    "current_policy",
    "reference_policy",
    "strict_policy",
    "critical_route_policy",
]
POLICY_LABELS = ["Current", "Reference", "Strict", "Critical-route"]
SCENARIOS = [
    "baseline",
    "high_direct_maturity",
    "high_recovery_efficiency",
    "lithium_aware_high_price",
    "combined_mitigation",
]
SCENARIO_LABELS = [
    "Baseline",
    "Direct\nmaturity",
    "High\nrecovery",
    "Li-aware\nhigh price",
    "Combined",
]
TECH_COLORS = {
    "Direct": "#4E79A7",
    "Hydro": "#59A14F",
    "Pyro": "#E15759",
}


def load_positions():
    countries = pd.read_csv(COUNTRY_FILE)
    countries = countries.dropna(subset=["iso3", "lat", "lon"]).copy()
    return countries.set_index("iso3")[["country", "lat", "lon"]]


def ordered_matrix(data, value_col):
    table = data.pivot_table(
        index="policy_scenario",
        columns="mitigation_scenario",
        values=value_col,
        aggfunc="mean",
    )
    return table.reindex(index=POLICIES, columns=SCENARIOS)


def draw_heatmap(ax, matrix, title, cmap, vmin=None, vmax=None, fmt="{:.0f}", cbar_label=""):
    image = ax.imshow(matrix.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, loc="left", fontsize=11, weight="bold")
    ax.set_xticks(np.arange(len(SCENARIOS)))
    ax.set_xticklabels(SCENARIO_LABELS, fontsize=8)
    ax.set_yticks(np.arange(len(POLICIES)))
    ax.set_yticklabels(POLICY_LABELS, fontsize=9)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.values[i, j]
            if pd.isna(value):
                label = ""
            else:
                label = fmt.format(value)
            color = "white" if value and value > np.nanmax(matrix.values) * 0.58 else "#111827"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)
    cbar = plt.colorbar(image, ax=ax, fraction=0.045, pad=0.02)
    cbar.ax.tick_params(labelsize=8)
    cbar.set_label(cbar_label, fontsize=8)


def plot_technology_mix(ax, routes):
    data = routes[
        (routes["year"] == YEAR)
        & (routes["policy_scenario"].isin(POLICIES))
        & (routes["mitigation_scenario"].isin(SCENARIOS))
        & (~routes["is_unprocessed"].astype(bool))
    ].copy()
    mix = (
        data.groupby(["mitigation_scenario", "technology"], as_index=False)[
            "recovered_lithium_t"
        ]
        .sum()
        .pivot_table(
            index="mitigation_scenario",
            columns="technology",
            values="recovered_lithium_t",
            fill_value=0.0,
        )
        .reindex(SCENARIOS)
        .fillna(0.0)
        / 1000.0
    )
    x = np.arange(len(SCENARIOS))
    bottom = np.zeros(len(SCENARIOS))
    for tech in ["Hydro", "Direct", "Pyro"]:
        if tech not in mix.columns:
            continue
        values = mix[tech].values
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=TECH_COLORS.get(tech, "#9CA3AF"),
            label=tech,
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += values
    ax.set_title("C. Technology pathway mix", loc="left", fontsize=11, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIO_LABELS, fontsize=8)
    ax.set_ylabel("Recovered Li (kt)")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def route_width(values, value):
    vmax = max(float(values.max()), 1.0)
    return 0.35 + 3.0 * (np.log10(abs(value) + 1) / np.log10(vmax + 1))


def draw_map(ax, routes, positions):
    data = routes[
        (routes["year"] == YEAR)
        & (routes["policy_scenario"] == "strict_policy")
        & (routes["mitigation_scenario"] == "combined_mitigation")
        & (~routes["is_unprocessed"].astype(bool))
    ].copy()
    table = (
        data.groupby(["source_iso3", "destination_iso3"], as_index=False)[
            "battery_embedded_secondary_li_t"
        ]
        .sum()
        .sort_values("battery_embedded_secondary_li_t", ascending=False)
        .head(22)
    )
    m = Basemap(
        projection="merc",
        llcrnrlon=-180,
        llcrnrlat=-60,
        urcrnrlon=180,
        urcrnrlat=75,
        resolution="c",
        suppress_ticks=True,
        ax=ax,
    )
    m.drawmapboundary(fill_color="white", linewidth=0)
    m.fillcontinents(color="#F3F4F6", lake_color="white", zorder=0)
    m.drawcoastlines(color="#D1D5DB", linewidth=0.25, zorder=1)
    m.drawcountries(color="#D1D5DB", linewidth=0.2, zorder=1)
    values = table["battery_embedded_secondary_li_t"]
    node_values = {}
    for _, row in table.iterrows():
        src = row["source_iso3"]
        dst = row["destination_iso3"]
        if src not in positions.index or dst not in positions.index:
            continue
        src_lon, src_lat = positions.loc[src, ["lon", "lat"]]
        dst_lon, dst_lat = positions.loc[dst, ["lon", "lat"]]
        x1, y1 = m(src_lon, src_lat)
        x2, y2 = m(dst_lon, dst_lat)
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="->",
                connectionstyle="arc3,rad=0.16",
                linewidth=route_width(values, row["battery_embedded_secondary_li_t"]),
                color="#F28E2B",
                alpha=0.58,
                shrinkA=1,
                shrinkB=1,
            ),
            zorder=4,
        )
        node_values[src] = node_values.get(src, 0.0) + row["battery_embedded_secondary_li_t"]
        node_values[dst] = node_values.get(dst, 0.0) + row["battery_embedded_secondary_li_t"]
    if node_values:
        max_node = max(node_values.values())
        for iso3, value in node_values.items():
            lon, lat = positions.loc[iso3, ["lon", "lat"]]
            x, y = m(lon, lat)
            size = 10 + 90 * np.sqrt(value / max_node)
            ax.scatter(
                x,
                y,
                s=size,
                facecolors="white",
                edgecolors="#111827",
                linewidths=0.5,
                alpha=0.9,
                zorder=5,
            )
    ax.set_title(
        "D. Representative optimized Li transfer\n(2050 strict + combined)",
        loc="left",
        fontsize=11,
        weight="bold",
    )
    handle = mlines.Line2D([], [], color="#F28E2B", linewidth=2.5, label="Li transfer route")
    ax.legend(handles=[handle], loc="lower left", frameon=False, fontsize=8)


def main():
    plt.rcParams["font.family"] = "Arial"
    summary = pd.read_csv(SUMMARY_FILE)
    routes = pd.read_csv(ROUTES_FILE)
    positions = load_positions()
    year_summary = summary[
        (summary["year"] == YEAR)
        & (summary["policy_scenario"].isin(POLICIES))
        & (summary["mitigation_scenario"].isin(SCENARIOS))
    ].copy()

    loss_matrix = ordered_matrix(year_summary, "loss_reduction_vs_baseline_pct")
    displacement_matrix = ordered_matrix(
        year_summary.assign(
            route_access_displaced_kt=lambda d: d["route_access_displaced_lithium_t"] / 1000.0
        ),
        "route_access_displaced_kt",
    )

    fig = plt.figure(figsize=(13.4, 8.2), dpi=300)
    grid = fig.add_gridspec(2, 2, wspace=0.26, hspace=0.36)
    ax_loss = fig.add_subplot(grid[0, 0])
    ax_route = fig.add_subplot(grid[0, 1])
    ax_mix = fig.add_subplot(grid[1, 0])
    ax_map = fig.add_subplot(grid[1, 1])

    draw_heatmap(
        ax_loss,
        loss_matrix,
        "A. Lithium-loss reduction potential",
        cmap="YlGnBu",
        vmin=0,
        vmax=70,
        fmt="{:.0f}",
        cbar_label="% vs baseline",
    )
    draw_heatmap(
        ax_route,
        displacement_matrix,
        "B. Route-access Li displacement",
        cmap="YlOrRd",
        vmin=0,
        fmt="{:.0f}",
        cbar_label="kt Li",
    )
    plot_technology_mix(ax_mix, routes)
    draw_map(ax_map, routes, positions)

    fig.suptitle(
        "Lithium-loss mitigation potential in policy-route-technology coupling",
        fontsize=15,
        weight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.015,
        "Matrices show 2050 policy-mitigation combinations. Route displacement is not equivalent to total lithium loss; loss reduction is driven by technology recovery and Direct maturity.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    png = OUTPUT_DIR / "policy_route_technology_loss_potential_figure.png"
    pdf = OUTPUT_DIR / "policy_route_technology_loss_potential_figure.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
