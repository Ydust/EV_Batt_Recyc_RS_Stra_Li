from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap


ROOT = Path(__file__).resolve().parent
ROUTES_FILE = (
    ROOT
    / "Figure_data"
    / "joint_policy_technology"
    / "joint_policy_transport_technology_routes_reference_relaxed_2030_2040_2050_with_open.csv"
)
SCENARIO_SUMMARY_FILE = (
    ROOT
    / "Figure_data"
    / "joint_policy_technology"
    / "lithium_loss_scenarios"
    / "lithium_loss_scenarios_summary.csv"
)
COUNTRY_FILE = ROOT / "all_countries.csv"
OUTPUT_DIR = ROOT / "Figure_data" / "joint_policy_technology"

VALUE_COL = "battery_embedded_secondary_li_t"
REFERENCE_POLICY = "route_access_open"
POLICIES = [
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
    "current_policy": "#4E79A7",
    "reference_policy": "#59A14F",
    "strict_policy": "#E15759",
    "critical_route_policy": "#B07AA1",
}
SCENARIO_LABELS = {
    "high_direct_maturity": "Direct maturity",
    "high_recovery_efficiency": "High recovery",
    "combined_mitigation": "Combined",
}
SCENARIO_COLORS = {
    "high_direct_maturity": "#4E79A7",
    "high_recovery_efficiency": "#59A14F",
    "combined_mitigation": "#F28E2B",
}


def load_positions():
    countries = pd.read_csv(COUNTRY_FILE)
    countries = countries.dropna(subset=["iso3", "lat", "lon"]).copy()
    return countries.set_index("iso3")[["country", "lat", "lon"]]


def prepare_baseline_tables(routes):
    real = routes[~routes["is_unprocessed"].astype(bool)].copy()
    policy_rows = []
    route_rows = []
    recovered_rows = []
    for year in sorted(real["year"].unique()):
        year_data = real[real["year"] == year]
        reference_total = float(
            year_data.loc[
                year_data["policy_scenario"] == REFERENCE_POLICY, "recovered_lithium_t"
            ].sum()
        )
        reference_routes = year_data[year_data["policy_scenario"] == REFERENCE_POLICY].groupby(
            ["source_iso3", "destination_iso3"], as_index=True
        )[VALUE_COL].sum()
        for policy in POLICIES:
            policy_data = year_data[year_data["policy_scenario"] == policy]
            policy_total = float(policy_data["recovered_lithium_t"].sum())
            recovered_rows.append(
                {
                    "year": int(year),
                    "policy_scenario": policy,
                    "recovered_lithium_ratio": (
                        policy_total / reference_total if reference_total else np.nan
                    ),
                }
            )
            policy_routes = policy_data.groupby(
                ["source_iso3", "destination_iso3"], as_index=True
            )[VALUE_COL].sum()
            comparison = pd.concat(
                [
                    reference_routes.rename("route_access_open_li_t"),
                    policy_routes.rename("policy_accessible_li_t"),
                ],
                axis=1,
            ).fillna(0.0)
            comparison["route_access_delta_li_t"] = (
                comparison["policy_accessible_li_t"]
                - comparison["route_access_open_li_t"]
            )
            comparison["route_displaced_li_t"] = (
                -comparison["route_access_delta_li_t"].clip(upper=0.0)
            )
            displacement = float(comparison["route_displaced_li_t"].sum())
            policy_rows.append(
                {
                    "year": int(year),
                    "policy_scenario": policy,
                    "route_displaced_kt_li": displacement / 1000.0,
                }
            )
            losses = comparison[comparison["route_displaced_li_t"] > 0].copy()
            losses = losses.reset_index()
            losses["year"] = int(year)
            losses["policy_scenario"] = policy
            route_rows.append(losses)
    return (
        pd.DataFrame(policy_rows),
        pd.concat(route_rows, ignore_index=True),
        pd.DataFrame(recovered_rows),
    )


def route_width(values, value):
    vmax = max(float(values.max()), 1.0)
    return 0.35 + 3.0 * (np.log10(abs(value) + 1) / np.log10(vmax + 1))


def draw_route_map(ax, by_route, positions):
    focus = by_route[
        (by_route["year"] == 2050)
        & (by_route["policy_scenario"] == "strict_policy")
    ].copy()
    focus = focus.sort_values("route_displaced_li_t", ascending=False).head(18)
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
    values = focus["route_displaced_li_t"]
    node_values = {}
    for _, row in focus.iterrows():
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
                connectionstyle="arc3,rad=0.18",
                linewidth=route_width(values, row["route_displaced_li_t"]),
                color="#F28E2B",
                alpha=0.62,
                shrinkA=1,
                shrinkB=1,
            ),
            zorder=4,
        )
        node_values[src] = node_values.get(src, 0.0) + row["route_displaced_li_t"]
        node_values[dst] = node_values.get(dst, 0.0) + row["route_displaced_li_t"]
    if node_values:
        max_node = max(node_values.values())
        for iso3, value in node_values.items():
            lon, lat = positions.loc[iso3, ["lon", "lat"]]
            x, y = m(lon, lat)
            size = 10 + 95 * np.sqrt(value / max_node)
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
        "A. Lithium-weighted route redistribution\n(2050 strict policy)",
        loc="left",
        fontsize=11,
        weight="bold",
    )
    handle = mlines.Line2D([], [], color="#F28E2B", linewidth=2.5, label="Lost benchmark route")
    ax.legend(handles=[handle], loc="lower left", frameon=False, fontsize=8)


def plot_displacement(ax, by_policy):
    years = sorted(by_policy["year"].unique())
    x = np.arange(len(years))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(POLICIES))
    for offset, policy in zip(offsets, POLICIES):
        subset = by_policy[by_policy["policy_scenario"] == policy].set_index("year")
        values = [subset.loc[year, "route_displaced_kt_li"] for year in years]
        ax.bar(
            x + offset,
            values,
            width=width,
            color=POLICY_COLORS[policy],
            label=POLICY_LABELS[policy],
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_title("B. Displaced benchmark Li routes", loc="left", fontsize=11, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(year) for year in years])
    ax.set_ylabel("Displaced Li (kt)")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")


def plot_recovered_ratio(ax, recovered):
    years = sorted(recovered["year"].unique())
    x = np.arange(len(years))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(POLICIES))
    for offset, policy in zip(offsets, POLICIES):
        subset = recovered[recovered["policy_scenario"] == policy].set_index("year")
        values = [(subset.loc[year, "recovered_lithium_ratio"] - 1.0) * 100.0 for year in years]
        ax.bar(
            x + offset,
            values,
            width=width,
            color=POLICY_COLORS[policy],
            edgecolor="white",
            linewidth=0.5,
        )
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_title("C. Total recovered Li vs open benchmark", loc="left", fontsize=11, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(year) for year in years])
    ax.set_ylabel("Change (%)")
    ax.set_ylim(-0.08, 0.08)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_scenarios(ax, summary):
    focus_policy = "strict_policy"
    focus = summary[
        summary["policy_scenario"].eq(focus_policy)
        & summary["mitigation_scenario"].isin(SCENARIO_LABELS)
    ].copy()
    years = sorted(focus["year"].unique())
    x = np.arange(len(years))
    width = 0.24
    scenarios = list(SCENARIO_LABELS)
    offsets = np.linspace(-width, width, len(scenarios))
    for offset, scenario in zip(offsets, scenarios):
        subset = focus[focus["mitigation_scenario"] == scenario].set_index("year")
        values = [subset.loc[year, "loss_reduction_vs_baseline_pct"] for year in years]
        ax.bar(
            x + offset,
            values,
            width=width,
            color=SCENARIO_COLORS[scenario],
            label=SCENARIO_LABELS[scenario],
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_title("D. Lithium-loss mitigation potential", loc="left", fontsize=11, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(year) for year in years])
    ax.set_ylabel("Loss reduction vs baseline (%)")
    ax.set_ylim(0, 78)
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")


def main():
    plt.rcParams["font.family"] = "Arial"
    routes = pd.read_csv(ROUTES_FILE)
    summary = pd.read_csv(SCENARIO_SUMMARY_FILE)
    by_policy, by_route, recovered = prepare_baseline_tables(routes)
    positions = load_positions()

    by_policy.to_csv(OUTPUT_DIR / "route_access_evidence_displacement.csv", index=False)
    recovered.to_csv(OUTPUT_DIR / "route_access_evidence_recovered_ratio.csv", index=False)

    fig = plt.figure(figsize=(13.2, 8.0), dpi=300)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], wspace=0.25, hspace=0.38)
    ax_map = fig.add_subplot(grid[0, 0])
    ax_displacement = fig.add_subplot(grid[0, 1])
    ax_recovered = fig.add_subplot(grid[1, 0])
    ax_scenarios = fig.add_subplot(grid[1, 1])

    draw_route_map(ax_map, by_route, positions)
    plot_displacement(ax_displacement, by_policy)
    plot_recovered_ratio(ax_recovered, recovered)
    plot_scenarios(ax_scenarios, summary)

    fig.suptitle(
        "Route-access constraints redistribute lithium flows, while mitigation is technology driven",
        fontsize=15,
        weight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.015,
        "Lithium quantities are battery-embedded secondary Li. Route displacement is measured against the route-access-unconstrained economic benchmark.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    png = OUTPUT_DIR / "route_access_loss_evidence_figure.png"
    pdf = OUTPUT_DIR / "route_access_loss_evidence_figure.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
