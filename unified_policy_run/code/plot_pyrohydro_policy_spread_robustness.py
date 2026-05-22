from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE_DIR = ROOT / "Figure_data" / "joint_policy_technology"
OUT_DIR = BASE_DIR / "pyrohydro_policy_robustness"
TREND_WINDOW_YEARS = 3
BASELINE_POLICY = "current_policy"
COMPARISON_POLICIES = [
    "reference_policy",
    "strict_policy",
    "critical_route_policy",
]
PLOT_POLICIES = [
    "current_policy",
    "reference_policy",
    "strict_policy",
    "critical_route_policy",
]
POLICY_ORDER = [
    "reference_policy",
    "current_policy",
    "strict_policy",
    "critical_route_policy",
]
POLICY_LABELS = {
    "current_policy": "Current",
    "reference_policy": "Reference",
    "strict_policy": "Strict",
    "critical_route_policy": "Critical-route",
}
TECHNOLOGY_ORDER = ["Direct", "Hydro", "PyroHydro"]
TECHNOLOGY_COLORS = {
    "Direct": "#0072B2",
    "Hydro": "#009E73",
    "PyroHydro": "#CC79A7",
}
CONTINENTS = ["Asia", "Europe", "America", "Africa", "Oceania"]
COUNTRY_FILE = ROOT / "all_countries.csv"
SCENARIOS = [
    {
        "scenario": "S1 conservative",
        "dir": BASE_DIR / "pyrohydro_sensitivity_conservative_annual_gurobi",
        "pyro_weight": 0.35,
        "developed": 0.85,
        "ev_producer": 0.90,
        "other": 1.00,
    },
    {
        "scenario": "S2 low-mid",
        "dir": BASE_DIR / "pyrohydro_sensitivity_s2_annual_gurobi",
        "pyro_weight": 0.30,
        "developed": 0.815,
        "ev_producer": 0.86,
        "other": 0.95,
    },
    {
        "scenario": "S3 medium",
        "dir": BASE_DIR / "pyrohydro_sensitivity_medium_annual_gurobi",
        "pyro_weight": 0.25,
        "developed": 0.78,
        "ev_producer": 0.82,
        "other": 0.90,
    },
    {
        "scenario": "S4 mid-high",
        "dir": BASE_DIR / "pyrohydro_sensitivity_s4_annual_gurobi",
        "pyro_weight": 0.20,
        "developed": 0.74,
        "ev_producer": 0.785,
        "other": 0.875,
    },
    {
        "scenario": "S5 strong",
        "dir": BASE_DIR / "pyrohydro_sensitivity_s5_annual_gurobi",
        "pyro_weight": 0.15,
        "developed": 0.70,
        "ev_producer": 0.75,
        "other": 0.85,
    },
]


# ---------------------------------------------------------------------------
# Continent-level share delta helpers (for radar charts)
# ---------------------------------------------------------------------------

def _country_continent_map():
    return pd.read_csv(COUNTRY_FILE).set_index("iso3")["continent"].to_dict()


def _read_routes_for_radar(scenario_label, scenario_dir, continent_map):
    routes = pd.read_csv(scenario_dir / "dynamic_scale_routes.csv")
    routes = routes[
        (routes["technology"].isin(TECHNOLOGY_ORDER))
        & (routes["policy_scenario"].isin([BASELINE_POLICY] + COMPARISON_POLICIES))
        & (~routes["is_unprocessed"].astype(bool))
    ].copy()
    routes["scenario"] = scenario_label
    routes["year"] = pd.to_numeric(routes["year"], errors="coerce").astype(int)
    routes["recovered_lithium_t"] = pd.to_numeric(routes["recovered_lithium_t"], errors="coerce").fillna(0.0)
    routes["continent"] = routes["destination_iso3"].map(continent_map).fillna("Other")
    routes = routes[routes["continent"].isin(CONTINENTS)]
    grouped = routes.groupby(
        ["scenario", "year", "policy_scenario", "continent", "technology"],
        as_index=False,
    ).agg(recovered_lithium_t=("recovered_lithium_t", "sum"))
    totals = grouped.groupby(
        ["scenario", "year", "policy_scenario", "continent"], as_index=False
    ).agg(total_recovered_lithium_t=("recovered_lithium_t", "sum"))
    grouped = grouped.merge(totals, on=["scenario", "year", "policy_scenario", "continent"], how="left")
    grouped["technology_share_pct"] = (
        grouped["recovered_lithium_t"] / grouped["total_recovered_lithium_t"] * 100.0
    ).fillna(0.0)
    return grouped


def build_share_delta_radar():
    continent_map = _country_continent_map()
    data = pd.concat(
        [_read_routes_for_radar(s["scenario"], s["dir"], continent_map) for s in SCENARIOS],
        ignore_index=True,
    )
    years = sorted(data["year"].unique())
    scenario_labels = [s["scenario"] for s in SCENARIOS]
    index = pd.MultiIndex.from_product(
        [scenario_labels, years, [BASELINE_POLICY] + COMPARISON_POLICIES, CONTINENTS, TECHNOLOGY_ORDER],
        names=["scenario", "year", "policy_scenario", "continent", "technology"],
    )
    data = (
        data.set_index(["scenario", "year", "policy_scenario", "continent", "technology"])
        .reindex(index, fill_value=0.0)
        .reset_index()
    )
    current = data[data["policy_scenario"] == BASELINE_POLICY][
        ["scenario", "year", "continent", "technology", "technology_share_pct"]
    ].rename(columns={"technology_share_pct": "current_share_pct"})
    delta = data.merge(current, on=["scenario", "year", "continent", "technology"], how="left")
    delta = delta[delta["policy_scenario"].isin(COMPARISON_POLICIES)].copy()
    delta["share_delta_vs_current_pp"] = delta["technology_share_pct"] - delta["current_share_pct"].fillna(0.0)
    return delta.groupby(["year", "policy_scenario", "continent", "technology"], as_index=False).agg(
        share_delta_vs_current_pp=("share_delta_vs_current_pp", "mean")
    )


def _plot_radar_combined(ax, share_delta, policy, limit):
    """All 3 technologies on one polar axes; dashed=2030, solid=2050. No fill, no background."""
    angles = np.linspace(0, 2 * np.pi, len(CONTINENTS), endpoint=False).tolist()
    angles += angles[:1]
    for technology in TECHNOLOGY_ORDER:
        subset = share_delta[
            (share_delta["technology"] == technology)
            & (share_delta["policy_scenario"] == policy)
            & (share_delta["year"].isin([2030, 2050]))
        ].copy()
        color = TECHNOLOGY_COLORS[technology]
        for year, alpha, lw, ls in [(2030, 0.60, 1.3, "--"), (2050, 0.95, 2.0, "-")]:
            values = [
                float(
                    subset.loc[
                        (subset["year"] == year) & (subset["continent"] == c),
                        "share_delta_vs_current_pp",
                    ].iloc[0]
                )
                if not subset[(subset["year"] == year) & (subset["continent"] == c)].empty
                else 0.0
                for c in CONTINENTS
            ]
            scaled = [0.5 + 0.5 * v / limit for v in values] + [0.5 + 0.5 * values[0] / limit]
            ax.plot(angles, scaled, color=color, alpha=alpha, linewidth=lw, linestyle=ls)
    ax.set_facecolor("none")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(CONTINENTS, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.grid(False)
    ring_theta = np.linspace(0, 2 * np.pi, 200)
    for r in [0.5, 1.0]:
        ax.plot(ring_theta, [r] * len(ring_theta),
                color="0.78", linewidth=0.8, zorder=1)
    ax.scatter([0], [0], color="0.78", s=8, zorder=1)
    for spine in ax.spines.values():
        spine.set_color("0.78")
        spine.set_linewidth(0.8)
    ax.tick_params(axis="x", pad=6)


# ---------------------------------------------------------------------------


def smooth_trend(frame, value_col):
    frame = frame.sort_values("year").copy()
    frame[value_col] = (
        frame[value_col]
        .astype(float)
        .rolling(TREND_WINDOW_YEARS, center=True, min_periods=1)
        .median()
    )
    return frame


def build_plot_data(data):
    smoothed = []
    for _, group in data.groupby(["policy_scenario", "technology"]):
        smoothed.append(smooth_trend(group, "recovered_lithium_t"))
    data = pd.concat(smoothed, ignore_index=True)
    totals = (
        data.groupby(["year", "policy_scenario"], as_index=False)["recovered_lithium_t"]
        .sum()
        .rename(columns={"recovered_lithium_t": "total_recovered_lithium_t"})
    )
    data = data.merge(totals, on=["year", "policy_scenario"], how="left")
    data["technology_share_pct"] = (
        data["recovered_lithium_t"] / data["total_recovered_lithium_t"] * 100.0
    ).fillna(0.0)
    return data


def load_all_scenarios():
    frames = []
    metadata_rows = []
    for scenario in SCENARIOS:
        data = pd.read_csv(scenario["dir"] / "dynamic_scale_summary.csv")
        for column in ["year", "recovered_lithium_t"]:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data = build_plot_data(data)
        data["scenario"] = scenario["scenario"]
        frames.append(data)
        metadata_rows.append(
            {
                key: value
                for key, value in scenario.items()
                if key not in {"dir"}
            }
        )
    return pd.concat(frames, ignore_index=True), pd.DataFrame(metadata_rows)


def build_policy_current_delta(data):
    rows = []
    for (scenario, year, technology), group in data.groupby(
        ["scenario", "year", "technology"]
    ):
        values = group.set_index("policy_scenario")["technology_share_pct"].reindex(
            POLICY_ORDER
        )
        current = float(values.get(BASELINE_POLICY, 0.0))
        for policy in COMPARISON_POLICIES:
            policy_share = float(values.get(policy, 0.0))
            rows.append(
                {
                    "scenario": scenario,
                    "year": int(year),
                    "technology": technology,
                    "policy_scenario": policy,
                    "current_pct": current,
                    "policy_pct": policy_share,
                    "technology_share_delta_vs_current_pp": policy_share - current,
                    "abs_technology_share_delta_vs_current_pp": abs(policy_share - current),
                }
            )
    return pd.DataFrame(rows)


def build_delta_band(delta):
    return (
        delta.groupby(["year", "technology", "policy_scenario"], as_index=False)
        .agg(
            mean_delta_vs_current_pp=("technology_share_delta_vs_current_pp", "mean"),
            min_delta_vs_current_pp=("technology_share_delta_vs_current_pp", "min"),
            max_delta_vs_current_pp=("technology_share_delta_vs_current_pp", "max"),
            mean_abs_delta_vs_current_pp=("abs_technology_share_delta_vs_current_pp", "mean"),
            max_abs_delta_vs_current_pp=("abs_technology_share_delta_vs_current_pp", "max"),
        )
    )


def build_pathway_band(data):
    return (
        data.groupby(["year", "technology", "policy_scenario"], as_index=False)
        .agg(
            recovered_li_mean_t=("recovered_lithium_t", "mean"),
            recovered_li_min_t=("recovered_lithium_t", "min"),
            recovered_li_max_t=("recovered_lithium_t", "max"),
            technology_share_mean_pct=("technology_share_pct", "mean"),
            technology_share_min_pct=("technology_share_pct", "min"),
            technology_share_max_pct=("technology_share_pct", "max"),
        )
    )


def build_summary(data, delta):
    rows = []
    for scenario, scenario_data in data.groupby("scenario"):
        for technology in TECHNOLOGY_ORDER:
            tech_data = scenario_data[scenario_data["technology"] == technology]
            tech_delta = delta[
                (delta["scenario"] == scenario) & (delta["technology"] == technology)
            ]
            rows.append(
                {
                    "scenario": scenario,
                    "technology": technology,
                    "mean_share_pct": tech_data["technology_share_pct"].mean(),
                    "max_share_pct": tech_data["technology_share_pct"].max(),
                    "mean_abs_policy_delta_vs_current_pp": tech_delta[
                        "abs_technology_share_delta_vs_current_pp"
                    ].mean(),
                    "max_abs_policy_delta_vs_current_pp": tech_delta[
                        "abs_technology_share_delta_vs_current_pp"
                    ].max(),
                }
            )
    return pd.DataFrame(rows)


def build_policy_effect_vs_current(data):
    rows = []
    for (scenario, year, technology), group in data.groupby(["scenario", "year", "technology"]):
        if technology not in TECHNOLOGY_ORDER:
            continue
        indexed = group.set_index("policy_scenario")
        if BASELINE_POLICY not in indexed.index:
            continue
        current = indexed.loc[BASELINE_POLICY]
        for policy in COMPARISON_POLICIES:
            if policy not in indexed.index:
                continue
            policy_row = indexed.loc[policy]
            rows.append(
                {
                    "scenario": scenario,
                    "year": int(year),
                    "technology": technology,
                    "policy_scenario": policy,
                    "recovered_li_delta_vs_current_t": float(
                        policy_row["recovered_lithium_t"] - current["recovered_lithium_t"]
                    ),
                    "technology_share_delta_vs_current_pp": float(
                        policy_row["technology_share_pct"] - current["technology_share_pct"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_policy_effect(effect):
    return (
        effect.groupby(["technology", "policy_scenario"], as_index=False)
        .agg(
            mean_recovered_li_delta_vs_current_t=("recovered_li_delta_vs_current_t", "mean"),
            max_abs_recovered_li_delta_vs_current_t=(
                "recovered_li_delta_vs_current_t",
                lambda series: series.abs().max(),
            ),
            mean_technology_share_delta_vs_current_pp=(
                "technology_share_delta_vs_current_pp",
                "mean",
            ),
            max_abs_technology_share_delta_vs_current_pp=(
                "technology_share_delta_vs_current_pp",
                lambda series: series.abs().max(),
            ),
        )
    )


def plot_policy_pathways(pathway_band):
    plt.rcParams.update({"font.family": "Arial"})
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(15.4, 6.2),
        sharex=True,
        dpi=300,
        gridspec_kw={"height_ratios": [1.5, 1.0], "hspace": 0.08, "wspace": 0.20},
    )
    metric_specs = [
        (
            "recovered_li",
            "Recovered Li (kt)",
            "recovered_li_mean_t",
            "recovered_li_min_t",
            "recovered_li_max_t",
            1000.0,
        ),
        (
            "share",
            "Technology share (%)",
            "technology_share_mean_pct",
            "technology_share_min_pct",
            "technology_share_max_pct",
            1.0,
        ),
    ]
    panel_labels = [["a", "b", "c", "d"], ["e", "f", "g", "h"]]
    for row_idx, (_, ylabel, mean_col, min_col, max_col, scale) in enumerate(metric_specs):
        for col_idx, policy in enumerate(PLOT_POLICIES):
            ax = axes[row_idx, col_idx]
            subset = pathway_band[pathway_band["policy_scenario"] == policy]
            for technology in TECHNOLOGY_ORDER:
                line = subset[subset["technology"] == technology].sort_values("year")
                if line.empty:
                    continue
                ax.plot(
                    line["year"],
                    line[mean_col] / scale,
                    color=TECHNOLOGY_COLORS[technology],
                    linestyle="-",
                    linewidth=2.25,
                    label=technology,
                )
                ax.fill_between(
                    line["year"],
                    line[min_col] / scale,
                    line[max_col] / scale,
                    color=TECHNOLOGY_COLORS[technology],
                    alpha=0.15,
                    linewidth=0,
                )
            ax.text(
                0.02,
                0.92,
                panel_labels[row_idx][col_idx],
                transform=ax.transAxes,
                fontsize=12,
                fontweight="bold",
                va="top",
            )
            if row_idx == 0:
                ax.set_title(POLICY_LABELS[policy], fontsize=12, pad=8)
            ax.set_xlim(2025, 2050)
            ax.set_xticks([2025, 2030, 2035, 2040, 2045, 2050])
            ax.grid(axis="y", color="0.88", linewidth=0.75)
            ax.tick_params(axis="both", labelsize=9, direction="in")
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel("")
            if row_idx == 1:
                ax.set_xlabel("Year")
            if "share" in mean_col:
                ax.set_ylim(0, 100)
                ax.set_yticks([0, 25, 50, 75, 100])
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
    handles = [
        Line2D([0], [0], color=TECHNOLOGY_COLORS[technology], lw=2.4, label=technology)
        for technology in TECHNOLOGY_ORDER
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.975))
    fig.suptitle(
        "Policy effects on technology pathways under S1-S5 PyroHydro sensitivity",
        y=1.035,
        fontsize=13,
    )
    fig.text(
        0.5,
        0.012,
        "Each column is one policy scenario. Colors distinguish technologies; lines show mean pathways across S1-S5 and shaded bands show the sensitivity range.",
        ha="center",
        fontsize=8.7,
        color="#374151",
    )
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.11, top=0.82)
    png = OUT_DIR / "policy_pathway_recovered_li_and_share_s1_s5.png"
    pdf = OUT_DIR / "policy_pathway_recovered_li_and_share_s1_s5.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


def plot_policy_pathways_with_share_heatmap(pathway_band, share_delta):
    plt.rcParams.update({"font.family": "Arial"})
    radar_limit = 8.0
    display_years = list(range(2030, 2051))
    fig = plt.figure(figsize=(15.4, 10.8), dpi=300)
    grid = fig.add_gridspec(
        3, 4,
        height_ratios=[1.55, 0.72, 1.25],
        hspace=0.20,
        wspace=0.22,
        left=0.06, right=0.985, bottom=0.10, top=0.88,
    )
    top_axes = [fig.add_subplot(grid[0, idx]) for idx in range(4)]
    heat_axes = [fig.add_subplot(grid[1, idx], sharex=top_axes[idx]) for idx in range(4)]
    # Row 2: col 0 = legend, cols 1-3 = radars
    radar_axes = [fig.add_subplot(grid[2, col + 1], projection="polar") for col in range(3)]

    years = [year for year in sorted(pathway_band["year"].dropna().astype(int).unique()) if year in display_years]
    panel_labels_top  = ["a", "b", "c", "d"]
    panel_labels_heat = ["e", "f", "g", "h"]
    panel_labels_radar = ["i", "j", "k"]
    heatmap_image = None

    # ---- row 0: recovered-Li line plots ----
    for col_idx, policy in enumerate(PLOT_POLICIES):
        policy_data = pathway_band[
            (pathway_band["policy_scenario"] == policy)
            & (pathway_band["year"].isin(display_years))
        ]
        ax = top_axes[col_idx]
        for technology in TECHNOLOGY_ORDER:
            line = policy_data[policy_data["technology"] == technology].sort_values("year")
            if line.empty:
                continue
            ax.plot(line["year"], line["recovered_li_mean_t"] / 1000.0,
                    color=TECHNOLOGY_COLORS[technology], linewidth=2.25, label=technology)
            ax.fill_between(line["year"],
                            line["recovered_li_min_t"] / 1000.0,
                            line["recovered_li_max_t"] / 1000.0,
                            color=TECHNOLOGY_COLORS[technology], alpha=0.14, linewidth=0)
        ax.set_title(POLICY_LABELS[policy], fontsize=12, pad=8)
        ax.text(0.02, 0.92, panel_labels_top[col_idx], transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")
        ax.set_xlim(2030, 2050)
        ax.set_xticks([2030, 2035, 2040, 2045, 2050])
        ax.grid(axis="y", color="0.88", linewidth=0.75)
        ax.tick_params(axis="both", labelsize=9, direction="in", labelbottom=False)
        if col_idx == 0:
            ax.set_ylabel("Recovered Li (kt)")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    # ---- row 1: technology-share heatmaps ----
    for col_idx, policy in enumerate(PLOT_POLICIES):
        policy_data = pathway_band[
            (pathway_band["policy_scenario"] == policy)
            & (pathway_band["year"].isin(display_years))
        ]
        heat_ax = heat_axes[col_idx]
        share_table = (
            policy_data.pivot_table(
                index="technology", columns="year",
                values="technology_share_mean_pct",
                aggfunc="first", fill_value=np.nan,
            ).reindex(index=TECHNOLOGY_ORDER, columns=years)
        )
        heatmap_image = heat_ax.imshow(
            share_table.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100,
            extent=[min(years) - 0.5, max(years) + 0.5, len(TECHNOLOGY_ORDER) - 0.5, -0.5],
        )
        heat_ax.text(0.02, 0.90, panel_labels_heat[col_idx], transform=heat_ax.transAxes,
                     fontsize=12, fontweight="bold", va="top", color="black")
        heat_ax.set_yticks(range(len(TECHNOLOGY_ORDER)))
        heat_ax.set_yticklabels(TECHNOLOGY_ORDER if col_idx == 0 else [])
        heat_ax.set_xticks([2030, 2035, 2040, 2045, 2050])
        heat_ax.set_xlabel("Year")
        heat_ax.tick_params(axis="both", labelsize=9, length=0)
        if col_idx == 0:
            heat_ax.set_ylabel("Technology share")
        for spine in heat_ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#374151")

    # ---- row 2: radars for comparison policies (cols 1-3) ----
    for col_idx, (ax, policy) in enumerate(zip(radar_axes, COMPARISON_POLICIES)):
        _plot_radar_combined(ax, share_delta, policy, radar_limit)
        ax.set_title(POLICY_LABELS[policy], fontsize=10, pad=10)
        ax.text(0.02, 1.12, panel_labels_radar[col_idx], transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")

    # ---- heatmap colorbar pinned to bottom-left ----
    cbar_ax = fig.add_axes([0.03, 0.215, 0.16, 0.018])
    cbar = fig.colorbar(heatmap_image, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Mean technology share across S1-S5 (%)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # ---- legend pinned to bottom-left corner of the figure (outside grid) ----
    tech_handles = [
        Line2D([0], [0], color=TECHNOLOGY_COLORS[t], lw=2.2, label=t)
        for t in TECHNOLOGY_ORDER
    ]
    year_handles = [
        Line2D([0], [0], color="#555555", lw=1.3, linestyle="--", label="Radar 2030"),
        Line2D([0], [0], color="#555555", lw=2.0, linestyle="-",  label="Radar 2050"),
    ]
    scale_handle = Patch(facecolor="none", edgecolor="none",
                         label=f"Radar: outer=+{radar_limit:.0f} pp, center=0, inner=\u2212{radar_limit:.0f} pp")
    # Place a small invisible axes in the bottom-left for the legend anchor
    ax_leg = fig.add_axes([0.01, 0.01, 0.20, 0.18])
    ax_leg.axis("off")
    ax_leg.legend(
        handles=tech_handles + year_handles + [scale_handle],
        loc="lower left",
        frameon=True,
        fontsize=7.8,
        title="Legend",
        title_fontsize=8.2,
        edgecolor="0.75",
        borderpad=0.6,
        labelspacing=0.35,
        handlelength=1.6,
    )

    fig.suptitle(
        "Policy effects on recovered lithium and technology mix under S1-S5 PyroHydro sensitivity",
        y=0.925, fontsize=13,
    )
    fig.text(
        0.5, 0.005,
        "Rows 1-2: recovered-Li pathways and technology-share heatmaps (columns = policies). "
        "Row 3: continent-level technology-share delta vs Current (pp), dashed=2030, solid=2050.",
        ha="center", fontsize=8.5, color="#374151",
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "policy_pathway_recovered_li_share_heatmap_s1_s5.png"
    pdf = OUT_DIR / "policy_pathway_recovered_li_share_heatmap_s1_s5.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


def plot_recovered_li_scatter_vs_current(pathway_band, data):
    plt.rcParams.update({"font.family": "Arial"})
    raw = data[data["technology"].isin(TECHNOLOGY_ORDER)].copy()
    cur_raw = (
        raw[raw["policy_scenario"] == BASELINE_POLICY]
        [["scenario", "year", "technology", "recovered_lithium_t"]]
        .rename(columns={"recovered_lithium_t": "current_li_t"})
    )
    cur_mean = (
        pathway_band[pathway_band["policy_scenario"] == BASELINE_POLICY]
        [["year", "technology", "recovered_li_mean_t"]]
        .rename(columns={"recovered_li_mean_t": "current_li_t"})
    )
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.2), dpi=300, sharex=True, sharey=True)
    panel_labels = ["a", "b", "c"]
    axis_max = max(
        cur_raw["current_li_t"].max(),
        raw[raw["policy_scenario"].isin(COMPARISON_POLICIES)]["recovered_lithium_t"].max(),
    ) / 1000.0 * 1.05

    key_years = [2040, 2050]
    year_markers = {2040: "^", 2050: "*"}
    year_sizes   = {2040: 90,  2050: 170}

    def _draw(ax, policy):
        other_raw = (
            raw[raw["policy_scenario"] == policy]
            [["scenario", "year", "technology", "recovered_lithium_t"]]
            .rename(columns={"recovered_lithium_t": "policy_li_t"})
        )
        merged_raw = cur_raw.merge(other_raw, on=["scenario", "year", "technology"], how="inner")
        other_mean = (
            pathway_band[pathway_band["policy_scenario"] == policy]
            [["year", "technology", "recovered_li_mean_t"]]
            .rename(columns={"recovered_li_mean_t": "policy_li_t"})
        )
        merged_mean = cur_mean.merge(other_mean, on=["year", "technology"], how="inner")
        for technology in TECHNOLOGY_ORDER:
            sub = merged_raw[merged_raw["technology"] == technology]
            if not sub.empty:
                ax.scatter(
                    sub["current_li_t"] / 1000.0,
                    sub["policy_li_t"] / 1000.0,
                    color=TECHNOLOGY_COLORS[technology],
                    alpha=0.18, s=14, edgecolor="none", linewidth=0,
                )
            sub_mean = merged_mean[merged_mean["technology"] == technology].sort_values("year")
            if sub_mean.empty:
                continue
            non_key = sub_mean[~sub_mean["year"].isin(key_years)]
            if not non_key.empty:
                ax.scatter(
                    non_key["current_li_t"] / 1000.0,
                    non_key["policy_li_t"] / 1000.0,
                    color=TECHNOLOGY_COLORS[technology],
                    s=22, marker="o", alpha=0.30,
                    edgecolor="none", linewidth=0, zorder=3,
                )
            key_pts = sub_mean[sub_mean["year"].isin(key_years)]
            for _, row in key_pts.iterrows():
                ax.scatter(
                    row["current_li_t"] / 1000.0,
                    row["policy_li_t"] / 1000.0,
                    color=TECHNOLOGY_COLORS[technology],
                    s=year_sizes[int(row["year"])],
                    marker=year_markers[int(row["year"])],
                    edgecolor="white", linewidth=0.8, zorder=4,
                )
        ax.plot([0, axis_max], [0, axis_max], color="0.5", linewidth=0.9, linestyle="--", zorder=1)
        return len(merged_raw), len(merged_mean)

    for ax, policy, panel in zip(axes, COMPARISON_POLICIES, panel_labels):
        _draw(ax, policy)
        ax.set_xlim(0, axis_max)
        ax.set_ylim(0, axis_max)
        ax.set_aspect("equal", adjustable="box")
        ax.text(0.02, 0.96, panel, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")
        ax.set_title(f"{POLICY_LABELS[policy]} vs Current", fontsize=11, pad=6)
        ax.set_xlabel("Current recovered Li (kt)")
        ax.grid(color="0.9", linewidth=0.6)
        ax.tick_params(axis="both", labelsize=9, direction="in")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("Policy recovered Li (kt)")

    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=TECHNOLOGY_COLORS[t],
               markeredgecolor="white", markersize=7, label=t)
        for t in TECHNOLOGY_ORDER
    ]
    style_handles = [
        Line2D([0], [0], marker="^", linestyle="", color="0.30",
               markeredgecolor="white", markersize=8, label="2040"),
        Line2D([0], [0], marker="*", linestyle="", color="0.30",
               markeredgecolor="white", markersize=13, label="2050"),
    ]
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.16, top=0.88, wspace=0.18)
    fig.legend(handles=handles + style_handles, loc="lower center", ncol=5,
               frameon=False, bbox_to_anchor=(0.5, 0.02), fontsize=9.5)
    fig.suptitle(
        "Recovered Li under each policy vs Current; key years 2040 and 2050",
        y=0.965, fontsize=12,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for policy, panel_ax, panel in zip(COMPARISON_POLICIES, axes, panel_labels):
        single_fig, single_ax = plt.subplots(figsize=(4.8, 4.6), dpi=300)
        _draw(single_ax, policy)
        single_ax.set_xlim(0, axis_max)
        single_ax.set_ylim(0, axis_max)
        single_ax.set_aspect("equal", adjustable="box")
        single_ax.set_xlabel("Current recovered Li (kt)")
        single_ax.set_ylabel(f"{POLICY_LABELS[policy]} recovered Li (kt)")
        single_ax.set_title(f"{POLICY_LABELS[policy]} vs Current (S1-S5)", fontsize=11)
        single_ax.grid(color="0.9", linewidth=0.6)
        single_ax.tick_params(axis="both", labelsize=9, direction="in")
        for spine in ["top", "right"]:
            single_ax.spines[spine].set_visible(False)
        single_ax.legend(frameon=False, fontsize=9, loc="lower right")
        single_fig.tight_layout()
        png = OUT_DIR / f"recovered_li_scatter_{policy}_vs_current.png"
        pdf = OUT_DIR / f"recovered_li_scatter_{policy}_vs_current.pdf"
        single_fig.savefig(png, dpi=220, bbox_inches="tight")
        single_fig.savefig(pdf, bbox_inches="tight")
        plt.close(single_fig)
        print(f"Wrote {png}")
        print(f"Wrote {pdf}")

    combined_png = OUT_DIR / "recovered_li_scatter_vs_current_combined.png"
    combined_pdf = OUT_DIR / "recovered_li_scatter_vs_current_combined.pdf"
    fig.savefig(combined_png, dpi=220)
    fig.savefig(combined_pdf)
    plt.close(fig)
    print(f"Wrote {combined_png}")
    print(f"Wrote {combined_pdf}")


def plot_scatter_radar_combined(pathway_band, data, share_delta):
    plt.rcParams.update({"font.family": "Arial"})
    raw = data[data["technology"].isin(TECHNOLOGY_ORDER)].copy()
    cur_raw = (
        raw[raw["policy_scenario"] == BASELINE_POLICY]
        [["scenario", "year", "technology", "recovered_lithium_t"]]
        .rename(columns={"recovered_lithium_t": "current_li_t"})
    )
    cur_mean = (
        pathway_band[pathway_band["policy_scenario"] == BASELINE_POLICY]
        [["year", "technology", "recovered_li_mean_t"]]
        .rename(columns={"recovered_li_mean_t": "current_li_t"})
    )
    axis_max = max(
        cur_raw["current_li_t"].max(),
        raw[raw["policy_scenario"].isin(COMPARISON_POLICIES)]["recovered_lithium_t"].max(),
    ) / 1000.0 * 1.05

    key_years = [2040, 2050]
    year_markers = {2040: "^", 2050: "*"}
    year_sizes   = {2040: 90,  2050: 170}
    radar_limit = 8.0

    fig = plt.figure(figsize=(13.2, 9.6), dpi=300)
    grid = fig.add_gridspec(
        2, 3, height_ratios=[1.0, 1.0],
        left=0.06, right=0.99, bottom=0.14, top=0.92,
        hspace=0.34, wspace=0.20,
    )
    scatter_axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    radar_axes = [fig.add_subplot(grid[1, i], projection="polar") for i in range(3)]

    panel_top = ["a", "b", "c"]
    panel_bot = ["d", "e", "f"]

    for ax, policy, panel in zip(scatter_axes, COMPARISON_POLICIES, panel_top):
        other_raw = (
            raw[raw["policy_scenario"] == policy]
            [["scenario", "year", "technology", "recovered_lithium_t"]]
            .rename(columns={"recovered_lithium_t": "policy_li_t"})
        )
        merged_raw = cur_raw.merge(other_raw, on=["scenario", "year", "technology"], how="inner")
        other_mean = (
            pathway_band[pathway_band["policy_scenario"] == policy]
            [["year", "technology", "recovered_li_mean_t"]]
            .rename(columns={"recovered_li_mean_t": "policy_li_t"})
        )
        merged_mean = cur_mean.merge(other_mean, on=["year", "technology"], how="inner")
        for technology in TECHNOLOGY_ORDER:
            sub = merged_raw[merged_raw["technology"] == technology]
            if not sub.empty:
                ax.scatter(
                    sub["current_li_t"] / 1000.0,
                    sub["policy_li_t"] / 1000.0,
                    color=TECHNOLOGY_COLORS[technology],
                    alpha=0.18, s=14, edgecolor="none", linewidth=0,
                )
            sub_mean = merged_mean[merged_mean["technology"] == technology].sort_values("year")
            if sub_mean.empty:
                continue
            non_key = sub_mean[~sub_mean["year"].isin(key_years)]
            if not non_key.empty:
                ax.scatter(
                    non_key["current_li_t"] / 1000.0,
                    non_key["policy_li_t"] / 1000.0,
                    color=TECHNOLOGY_COLORS[technology],
                    s=22, marker="o", alpha=0.30,
                    edgecolor="none", linewidth=0, zorder=3,
                )
            for _, row in sub_mean[sub_mean["year"].isin(key_years)].iterrows():
                ax.scatter(
                    row["current_li_t"] / 1000.0,
                    row["policy_li_t"] / 1000.0,
                    color=TECHNOLOGY_COLORS[technology],
                    s=year_sizes[int(row["year"])],
                    marker=year_markers[int(row["year"])],
                    edgecolor="white", linewidth=0.8, zorder=4,
                )
        ax.plot([0, axis_max], [0, axis_max], color="0.5", linewidth=0.9, linestyle="--", zorder=1)
        ax.set_xlim(0, axis_max)
        ax.set_ylim(0, axis_max)
        ax.set_aspect("equal", adjustable="box")
        ax.text(0.02, 0.96, panel, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")
        ax.set_title(f"{POLICY_LABELS[policy]} vs Current", fontsize=11, pad=6)
        ax.set_xlabel("Current recovered Li (kt)")
        ax.grid(color="0.9", linewidth=0.6)
        ax.tick_params(axis="both", labelsize=9, direction="in")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
    scatter_axes[0].set_ylabel("Policy recovered Li (kt)")

    for ax, policy, panel in zip(radar_axes, COMPARISON_POLICIES, panel_bot):
        _plot_radar_combined(ax, share_delta, policy, radar_limit)
        ax.set_title(POLICY_LABELS[policy], fontsize=11, pad=18)
        ax.text(-0.10, 1.14, panel, transform=ax.transAxes,
                fontsize=12, fontweight="bold", va="top")
        bbox = ax.get_position()
        cx, cy = bbox.x0 + bbox.width / 2, bbox.y0 + bbox.height / 2
        new_w, new_h = bbox.width * 0.72, bbox.height * 0.72
        ax.set_position([cx - new_w / 2, cy - new_h / 2, new_w, new_h])

    tech_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=TECHNOLOGY_COLORS[t],
               markeredgecolor="white", markersize=7, label=t)
        for t in TECHNOLOGY_ORDER
    ] + [
        Line2D([0], [0], marker="^", linestyle="", color="0.30",
               markeredgecolor="white", markersize=8, label="2040"),
        Line2D([0], [0], marker="*", linestyle="", color="0.30",
               markeredgecolor="white", markersize=13, label="2050"),
    ]
    radar_handles = [
        Line2D([0], [0], color="0.30", linewidth=1.3, linestyle="--",
               label="Continent share Δ vs Current, 2030 (pp)"),
        Line2D([0], [0], color="0.30", linewidth=2.0, linestyle="-",
               label="Continent share Δ vs Current, 2050 (pp)"),
    ]
    leg1 = fig.legend(handles=tech_handles, loc="lower center", ncol=5,
                      frameon=False, bbox_to_anchor=(0.5, 0.055), fontsize=9.2)
    fig.add_artist(leg1)
    fig.legend(handles=radar_handles, loc="lower center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 0.005), fontsize=9.2)
    fig.suptitle(
        "Policy effects: recovered Li (top) and continent technology-share delta vs Current (bottom)",
        y=0.965, fontsize=12,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "scatter_radar_policy_vs_current_combined.png"
    pdf = OUT_DIR / "scatter_radar_policy_vs_current_combined.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


def plot_policy_current_delta(delta_band):
    plt.rcParams.update({"font.family": "Arial"})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True, dpi=300)
    for ax, technology, panel in zip(axes, TECHNOLOGY_ORDER, ["a", "b", "c"]):
        subset = delta_band[delta_band["technology"] == technology]
        for policy in COMPARISON_POLICIES:
            line = subset[subset["policy_scenario"] == policy].sort_values("year")
            ax.plot(
                line["year"],
                line["mean_delta_vs_current_pp"],
                color=POLICY_COLORS[policy],
                linestyle="-",
                linewidth=2.2,
                label=f"{POLICY_LABELS[policy]} - Current",
            )
            ax.fill_between(
                line["year"],
                line["min_delta_vs_current_pp"],
                line["max_delta_vs_current_pp"],
                color=POLICY_COLORS[policy],
                alpha=0.15,
                linewidth=0,
            )
        ax.axhline(0, color="0.25", linewidth=0.8)
        ax.text(0.02, 0.95, panel, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
        ax.set_title(technology)
        ax.set_xlim(2025, 2050)
        ax.set_xticks([2025, 2030, 2035, 2040, 2045, 2050])
        ax.grid(axis="y", color="0.88", linewidth=0.8)
        ax.set_xlabel("Year")
    axes[0].set_ylabel("Technology-share change vs Current (percentage points)")
    handles = [
        Line2D([0], [0], color=POLICY_COLORS[policy], lw=2.2, label=f"{POLICY_LABELS[policy]} - Current")
        for policy in COMPARISON_POLICIES
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.suptitle(
        "Policy impacts on technology paths relative to Current; shaded ranges are S1-S5 PyroHydro sensitivity",
        y=1.16,
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = OUT_DIR / "policy_delta_vs_current_by_technology_s1_s5.png"
    pdf = OUT_DIR / "policy_delta_vs_current_by_technology_s1_s5.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data, metadata = load_all_scenarios()
    delta = build_policy_current_delta(data)
    delta_band = build_delta_band(delta)
    pathway_band = build_pathway_band(data)
    effect = build_policy_effect_vs_current(data)
    effect_summary = summarize_policy_effect(effect)
    summary = build_summary(data, delta)
    data.to_csv(OUT_DIR / "pyrohydro_s1_s5_trend_corrected.csv", index=False)
    pathway_band.to_csv(OUT_DIR / "pyrohydro_s1_s5_policy_pathway_band.csv", index=False)
    effect.to_csv(OUT_DIR / "pyrohydro_s1_s5_policy_effect_vs_current.csv", index=False)
    effect_summary.to_csv(OUT_DIR / "pyrohydro_s1_s5_policy_effect_vs_current_summary.csv", index=False)
    delta.to_csv(OUT_DIR / "pyrohydro_s1_s5_policy_delta_vs_current.csv", index=False)
    delta_band.to_csv(OUT_DIR / "pyrohydro_s1_s5_policy_delta_vs_current_band.csv", index=False)
    summary.to_csv(OUT_DIR / "pyrohydro_s1_s5_policy_delta_vs_current_summary.csv", index=False)
    metadata.to_csv(OUT_DIR / "pyrohydro_s1_s5_parameters.csv", index=False)
    plot_policy_pathways(pathway_band)
    share_delta = build_share_delta_radar()
    share_delta.to_csv(OUT_DIR / "pyrohydro_s1_s5_policy_continent_share_delta.csv", index=False)
    plot_policy_pathways_with_share_heatmap(pathway_band, share_delta)
    plot_recovered_li_scatter_vs_current(pathway_band, data)
    plot_scatter_radar_combined(pathway_band, data, share_delta)


if __name__ == "__main__":
    main()
