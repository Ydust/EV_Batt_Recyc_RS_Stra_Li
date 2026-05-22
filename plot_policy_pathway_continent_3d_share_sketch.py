from pathlib import Path

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE_DIR = ROOT / "Figure_data" / "joint_policy_technology"
OUT_DIR = BASE_DIR / "pyrohydro_policy_robustness"
COUNTRY_FILE = ROOT / "all_countries.csv"
SCENARIOS = [
    ("S1", BASE_DIR / "pyrohydro_sensitivity_conservative_annual_gurobi"),
    ("S2", BASE_DIR / "pyrohydro_sensitivity_s2_annual_gurobi"),
    ("S3", BASE_DIR / "pyrohydro_sensitivity_medium_annual_gurobi"),
    ("S4", BASE_DIR / "pyrohydro_sensitivity_s4_annual_gurobi"),
    ("S5", BASE_DIR / "pyrohydro_sensitivity_s5_annual_gurobi"),
]
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
TECHNOLOGIES = ["Direct", "Hydro", "PyroHydro"]
TECHNOLOGY_COLORS = {
    "Direct": "#0072B2",
    "Hydro": "#009E73",
    "PyroHydro": "#CC79A7",
}
CONTINENTS = ["Asia", "Europe", "America", "Africa", "Oceania"]


def load_country_continents():
    countries = pd.read_csv(COUNTRY_FILE)
    return countries.set_index("iso3")["continent"].to_dict()


def smooth_trend(frame, value_col):
    frame = frame.sort_values("year").copy()
    frame[value_col] = (
        frame[value_col]
        .astype(float)
        .rolling(3, center=True, min_periods=1)
        .median()
    )
    return frame


def read_summary_scenario(label, data_dir):
    data = pd.read_csv(data_dir / "dynamic_scale_summary.csv")
    data = data[data["technology"].isin(TECHNOLOGIES)].copy()
    for column in ["year", "recovered_lithium_t"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    smoothed = []
    for _, group in data.groupby(["policy_scenario", "technology"]):
        smoothed.append(smooth_trend(group, "recovered_lithium_t"))
    data = pd.concat(smoothed, ignore_index=True)
    data["scenario"] = label
    return data


def build_global_band():
    data = pd.concat(
        [read_summary_scenario(label, path) for label, path in SCENARIOS],
        ignore_index=True,
    )
    return data.groupby(["year", "policy_scenario", "technology"], as_index=False).agg(
        recovered_li_mean_t=("recovered_lithium_t", "mean"),
        recovered_li_min_t=("recovered_lithium_t", "min"),
        recovered_li_max_t=("recovered_lithium_t", "max"),
    )


def read_routes_scenario(label, data_dir, continent_map):
    routes = pd.read_csv(data_dir / "dynamic_scale_routes.csv")
    routes = routes[
        (routes["policy_scenario"].isin(POLICIES))
        & (routes["technology"].isin(TECHNOLOGIES))
        & (~routes["is_unprocessed"].astype(bool))
    ].copy()
    routes["year"] = pd.to_numeric(routes["year"], errors="coerce").astype(int)
    routes["recovered_lithium_t"] = pd.to_numeric(
        routes["recovered_lithium_t"], errors="coerce"
    ).fillna(0.0)
    routes["continent"] = routes["destination_iso3"].map(continent_map).fillna("Other")
    routes = routes[routes["continent"].isin(CONTINENTS)]
    grouped = routes.groupby(
        ["year", "policy_scenario", "continent", "technology"],
        as_index=False,
    ).agg(recovered_lithium_t=("recovered_lithium_t", "sum"))
    totals = grouped.groupby(["year", "policy_scenario", "continent"], as_index=False).agg(
        total_recovered_lithium_t=("recovered_lithium_t", "sum")
    )
    grouped = grouped.merge(totals, on=["year", "policy_scenario", "continent"], how="left")
    grouped["technology_share_pct"] = (
        grouped["recovered_lithium_t"] / grouped["total_recovered_lithium_t"] * 100.0
    ).fillna(0.0)
    grouped["scenario"] = label
    return grouped


def build_continent_share_band():
    continent_map = load_country_continents()
    data = pd.concat(
        [read_routes_scenario(label, path, continent_map) for label, path in SCENARIOS],
        ignore_index=True,
    )
    years = sorted(data["year"].unique())
    index = pd.MultiIndex.from_product(
        [[label for label, _ in SCENARIOS], years, POLICIES, CONTINENTS, TECHNOLOGIES],
        names=["scenario", "year", "policy_scenario", "continent", "technology"],
    )
    data = (
        data.set_index(["scenario", "year", "policy_scenario", "continent", "technology"])
        .reindex(index, fill_value=0.0)
        .reset_index()
    )
    return data.groupby(["year", "policy_scenario", "continent", "technology"], as_index=False).agg(
        mean_share_pct=("technology_share_pct", "mean"),
        min_share_pct=("technology_share_pct", "min"),
        max_share_pct=("technology_share_pct", "max"),
    )


def build_continent_share_delta_vs_current(continent_band):
    current = continent_band[continent_band["policy_scenario"] == "current_policy"][
        ["year", "continent", "technology", "mean_share_pct"]
    ].rename(columns={"mean_share_pct": "current_mean_share_pct"})
    delta = continent_band.merge(current, on=["year", "continent", "technology"], how="left")
    delta = delta[delta["policy_scenario"] != "current_policy"].copy()
    delta["share_delta_vs_current_pp"] = (
        delta["mean_share_pct"] - delta["current_mean_share_pct"].fillna(0.0)
    )
    return delta


def format_top_axis(ax, col_idx):
    ax.set_xlim(2025, 2050)
    ax.set_xticks([2025, 2030, 2035, 2040, 2045, 2050])
    ax.tick_params(axis="both", labelsize=8.5, direction="in", labelbottom=False)
    ax.grid(axis="y", color="0.88", linewidth=0.7)
    if col_idx == 0:
        ax.set_ylabel("Recovered Li (kt)", fontsize=9.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_sketch(global_band, continent_band):
    plt.rcParams.update({"font.family": "Arial"})
    fig = plt.figure(figsize=(15.6, 7.3), dpi=300)
    grid = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.0, 1.1],
        hspace=0.08,
        wspace=0.10,
    )
    top_axes = [fig.add_subplot(grid[0, idx]) for idx in range(4)]
    bottom_axes = [fig.add_subplot(grid[1, idx], projection="3d") for idx in range(4)]
    panel_labels = ["a", "b", "c", "d", "e", "f", "g", "h"]

    for col_idx, policy in enumerate(POLICIES):
        ax = top_axes[col_idx]
        policy_data = global_band[global_band["policy_scenario"] == policy]
        for technology in TECHNOLOGIES:
            line = policy_data[policy_data["technology"] == technology].sort_values("year")
            ax.plot(
                line["year"],
                line["recovered_li_mean_t"] / 1000.0,
                color=TECHNOLOGY_COLORS[technology],
                linewidth=2.1,
            )
            ax.fill_between(
                line["year"],
                line["recovered_li_min_t"] / 1000.0,
                line["recovered_li_max_t"] / 1000.0,
                color=TECHNOLOGY_COLORS[technology],
                alpha=0.13,
                linewidth=0,
            )
        ax.set_title(POLICY_LABELS[policy], fontsize=12, pad=8)
        ax.text(
            0.02,
            0.90,
            panel_labels[col_idx],
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )
        format_top_axis(ax, col_idx)

        ax3d = bottom_axes[col_idx]
        share_data = continent_band[continent_band["policy_scenario"] == policy]
        years = sorted(share_data["year"].unique())
        x_grid, y_grid = np.meshgrid(years, range(len(CONTINENTS)))
        for technology in TECHNOLOGIES:
            tech_data = share_data[share_data["technology"] == technology]
            surface = (
                tech_data.pivot_table(
                    index="continent",
                    columns="year",
                    values="mean_share_pct",
                    aggfunc="first",
                    fill_value=0.0,
                )
                .reindex(index=CONTINENTS, columns=years, fill_value=0.0)
            )
            ax3d.plot_surface(
                x_grid,
                y_grid,
                surface.values,
                color=TECHNOLOGY_COLORS[technology],
                alpha=0.38,
                linewidth=0.25,
                edgecolor=TECHNOLOGY_COLORS[technology],
                antialiased=True,
                shade=False,
            )
        ax3d.text2D(
            0.02,
            0.92,
            panel_labels[col_idx + 4],
            transform=ax3d.transAxes,
            fontsize=11,
            fontweight="bold",
        )
        ax3d.set_xlim(2025, 2050)
        ax3d.set_ylim(-0.3, len(CONTINENTS) - 0.7)
        ax3d.set_zlim(0, 100)
        ax3d.set_xticks([2025, 2035, 2045, 2050])
        ax3d.set_yticks(range(len(CONTINENTS)))
        ax3d.set_yticklabels(CONTINENTS if col_idx == 0 else [], fontsize=7.2)
        ax3d.set_zticks([0, 50, 100])
        ax3d.set_xlabel("Year", labelpad=3, fontsize=8)
        if col_idx == 0:
            ax3d.set_ylabel("Continent", labelpad=5, fontsize=8)
        ax3d.set_zlabel("Share (%)", labelpad=4, fontsize=8)
        ax3d.tick_params(axis="both", labelsize=7)
        ax3d.view_init(elev=23, azim=-58)
        ax3d.grid(True)

    handles = [
        Line2D([0], [0], color=TECHNOLOGY_COLORS[technology], lw=2.4, label=technology)
        for technology in TECHNOLOGIES
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.975))
    fig.suptitle(
        "Policy pathways with continent-disaggregated technology shares",
        fontsize=13,
        y=1.02,
    )
    fig.text(
        0.5,
        0.02,
        "Top panels show global recovered lithium with S1-S5 ranges. Bottom 3D panels show mean technology share by destination continent.",
        ha="center",
        fontsize=8.6,
        color="#374151",
    )
    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.09, top=0.86)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "policy_pathway_continent_3d_share_sketch.png"
    pdf = OUT_DIR / "policy_pathway_continent_3d_share_sketch.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


def plot_delta_surface_sketch(global_band, continent_delta):
    plt.rcParams.update({"font.family": "Arial"})
    comparison_policies = ["reference_policy", "strict_policy", "critical_route_policy"]
    fig = plt.figure(figsize=(14.2, 7.15), dpi=300)
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 1.1],
        hspace=0.08,
        wspace=0.08,
    )
    top_axes = [fig.add_subplot(grid[0, idx]) for idx in range(3)]
    bottom_axes = [fig.add_subplot(grid[1, idx], projection="3d") for idx in range(3)]
    panel_labels = ["a", "b", "c", "d", "e", "f"]

    for col_idx, policy in enumerate(comparison_policies):
        ax = top_axes[col_idx]
        for technology in TECHNOLOGIES:
            current_line = global_band[
                (global_band["policy_scenario"] == "current_policy")
                & (global_band["technology"] == technology)
            ].sort_values("year")
            policy_line = global_band[
                (global_band["policy_scenario"] == policy)
                & (global_band["technology"] == technology)
            ].sort_values("year")
            if current_line.empty or policy_line.empty:
                continue
            merged = policy_line[["year", "recovered_li_mean_t"]].merge(
                current_line[["year", "recovered_li_mean_t"]],
                on="year",
                suffixes=("_policy", "_current"),
            )
            ax.plot(
                merged["year"],
                (merged["recovered_li_mean_t_policy"] - merged["recovered_li_mean_t_current"]) / 1000.0,
                color=TECHNOLOGY_COLORS[technology],
                linewidth=2.1,
            )
        ax.axhline(0, color="#111827", linewidth=0.8)
        ax.set_title(f"{POLICY_LABELS[policy]} - Current", fontsize=12, pad=8)
        ax.text(
            0.02,
            0.90,
            panel_labels[col_idx],
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
        )
        ax.set_xlim(2025, 2050)
        ax.set_xticks([2025, 2030, 2035, 2040, 2045, 2050])
        ax.tick_params(axis="both", labelsize=8.5, direction="in", labelbottom=False)
        ax.grid(axis="y", color="0.88", linewidth=0.7)
        if col_idx == 0:
            ax.set_ylabel("Recovered Li delta (kt)", fontsize=9.5)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        ax3d = bottom_axes[col_idx]
        share_data = continent_delta[continent_delta["policy_scenario"] == policy]
        years = sorted(share_data["year"].unique())
        tech_offsets = {technology: idx * (len(CONTINENTS) + 1.0) for idx, technology in enumerate(TECHNOLOGIES)}
        for technology in TECHNOLOGIES:
            tech_data = share_data[share_data["technology"] == technology]
            surface = (
                tech_data.pivot_table(
                    index="continent",
                    columns="year",
                    values="share_delta_vs_current_pp",
                    aggfunc="first",
                    fill_value=0.0,
                )
                .reindex(index=CONTINENTS, columns=years, fill_value=0.0)
            )
            x_grid, y_grid = np.meshgrid(years, np.arange(len(CONTINENTS)) + tech_offsets[technology])
            ax3d.plot_surface(
                x_grid,
                y_grid,
                surface.values,
                color=TECHNOLOGY_COLORS[technology],
                alpha=0.48,
                linewidth=0.25,
                edgecolor=TECHNOLOGY_COLORS[technology],
                antialiased=True,
                shade=False,
            )
            ax3d.plot_wireframe(
                x_grid,
                y_grid,
                np.zeros_like(surface.values),
                color="#9CA3AF",
                linewidth=0.25,
                alpha=0.45,
            )
        y_ticks = []
        y_labels = []
        for technology in TECHNOLOGIES:
            offset = tech_offsets[technology]
            for continent_idx, continent in enumerate(CONTINENTS):
                y_ticks.append(offset + continent_idx)
                y_labels.append(f"{technology[:1]}-{continent}" if col_idx == 0 else "")
        ax3d.text2D(
            0.02,
            0.92,
            panel_labels[col_idx + 3],
            transform=ax3d.transAxes,
            fontsize=11,
            fontweight="bold",
        )
        ax3d.set_xlim(2025, 2050)
        ax3d.set_ylim(-0.3, max(y_ticks) + 0.5)
        ax3d.set_zlim(-45, 45)
        ax3d.set_xticks([2025, 2035, 2045, 2050])
        ax3d.set_yticks(y_ticks)
        ax3d.set_yticklabels(y_labels, fontsize=5.8)
        ax3d.set_zticks([-40, 0, 40])
        ax3d.set_xlabel("Year", labelpad=3, fontsize=8)
        if col_idx == 0:
            ax3d.set_ylabel("Technology-continent", labelpad=5, fontsize=8)
        ax3d.set_zlabel("Share delta (pp)", labelpad=4, fontsize=8)
        ax3d.tick_params(axis="both", labelsize=7)
        ax3d.view_init(elev=24, azim=-57)
        ax3d.grid(True)

    handles = [
        Line2D([0], [0], color=TECHNOLOGY_COLORS[technology], lw=2.4, label=technology)
        for technology in TECHNOLOGIES
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.975))
    fig.suptitle(
        "Policy impact surfaces relative to Current",
        fontsize=13,
        y=1.02,
    )
    fig.text(
        0.5,
        0.02,
        "Top panels show global recovered-Li deltas. Bottom surfaces show technology-share changes by destination continent; zero planes are grey.",
        ha="center",
        fontsize=8.6,
        color="#374151",
    )
    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.09, top=0.86)
    png = OUT_DIR / "policy_impact_continent_3d_share_delta_vs_current.png"
    pdf = OUT_DIR / "policy_impact_continent_3d_share_delta_vs_current.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


def main():
    global_band = build_global_band()
    continent_band = build_continent_share_band()
    continent_delta = build_continent_share_delta_vs_current(continent_band)
    continent_band.to_csv(OUT_DIR / "policy_pathway_continent_3d_share_sketch.csv", index=False)
    continent_delta.to_csv(OUT_DIR / "policy_impact_continent_share_delta_vs_current.csv", index=False)
    plot_sketch(global_band, continent_band)
    plot_delta_surface_sketch(global_band, continent_delta)


if __name__ == "__main__":
    main()
