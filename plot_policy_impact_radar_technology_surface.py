from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm


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
BASELINE_POLICY = "current_policy"
COMPARISON_POLICIES = ["reference_policy", "strict_policy", "critical_route_policy"]
KEY_YEARS = [2030, 2040, 2050]
POLICY_LABELS = {
    "reference_policy": "Reference - Current",
    "strict_policy": "Strict - Current",
    "critical_route_policy": "Critical-route - Current",
}
POLICY_COLORS = {
    "reference_policy": "#4E79A7",
    "strict_policy": "#E15759",
    "critical_route_policy": "#59A14F",
}
TECHNOLOGIES = ["Direct", "Hydro", "PyroHydro"]
TECHNOLOGY_LABELS = {"Direct": "Direct", "Hydro": "Hydro", "PyroHydro": "PyroHydro"}
CONTINENTS = ["Asia", "Europe", "America", "Africa", "Oceania"]


def country_continent_map():
    return pd.read_csv(COUNTRY_FILE).set_index("iso3")["continent"].to_dict()


def read_summary(label, data_dir):
    data = pd.read_csv(data_dir / "dynamic_scale_summary.csv")
    data = data[data["technology"].isin(TECHNOLOGIES)].copy()
    data["scenario"] = label
    for column in ["year", "recovered_lithium_t"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def build_global_delta():
    data = pd.concat([read_summary(label, path) for label, path in SCENARIOS], ignore_index=True)
    data = data.groupby(["scenario", "year", "policy_scenario", "technology"], as_index=False).agg(
        recovered_lithium_t=("recovered_lithium_t", "sum")
    )
    current = data[data["policy_scenario"] == BASELINE_POLICY][
        ["scenario", "year", "technology", "recovered_lithium_t"]
    ].rename(columns={"recovered_lithium_t": "current_recovered_lithium_t"})
    delta = data.merge(current, on=["scenario", "year", "technology"], how="left")
    delta = delta[delta["policy_scenario"].isin(COMPARISON_POLICIES)].copy()
    delta["recovered_li_delta_kt"] = (
        delta["recovered_lithium_t"] - delta["current_recovered_lithium_t"]
    ) / 1000.0
    return delta.groupby(["year", "policy_scenario", "technology"], as_index=False).agg(
        recovered_li_delta_kt=("recovered_li_delta_kt", "mean")
    )


def build_global_pct_change():
    """Per-scenario % change of recovered Li vs current_policy, aggregated to mean/min/max."""
    data = pd.concat([read_summary(label, path) for label, path in SCENARIOS], ignore_index=True)
    data = data.groupby(["scenario", "year", "policy_scenario", "technology"], as_index=False).agg(
        recovered_lithium_t=("recovered_lithium_t", "sum")
    )
    current = data[data["policy_scenario"] == BASELINE_POLICY][
        ["scenario", "year", "technology", "recovered_lithium_t"]
    ].rename(columns={"recovered_lithium_t": "current_recovered_lithium_t"})
    df = data.merge(current, on=["scenario", "year", "technology"], how="left")
    df = df[df["policy_scenario"].isin(COMPARISON_POLICIES)].copy()
    df["pct_change"] = (
        (df["recovered_lithium_t"] - df["current_recovered_lithium_t"])
        / df["current_recovered_lithium_t"].replace(0.0, np.nan)
        * 100.0
    )
    return df.groupby(["year", "policy_scenario", "technology"], as_index=False).agg(
        pct_mean=("pct_change", "mean"),
        pct_min=("pct_change", "min"),
        pct_max=("pct_change", "max"),
    )


def read_routes(label, data_dir, continent_map):
    routes = pd.read_csv(data_dir / "dynamic_scale_routes.csv")
    routes = routes[
        (routes["technology"].isin(TECHNOLOGIES))
        & (routes["policy_scenario"].isin([BASELINE_POLICY] + COMPARISON_POLICIES))
        & (~routes["is_unprocessed"].astype(bool))
    ].copy()
    routes["scenario"] = label
    routes["year"] = pd.to_numeric(routes["year"], errors="coerce").astype(int)
    routes["recovered_lithium_t"] = pd.to_numeric(
        routes["recovered_lithium_t"], errors="coerce"
    ).fillna(0.0)
    routes["continent"] = routes["destination_iso3"].map(continent_map).fillna("Other")
    routes = routes[routes["continent"].isin(CONTINENTS)]
    grouped = routes.groupby(
        ["scenario", "year", "policy_scenario", "continent", "technology"],
        as_index=False,
    ).agg(recovered_lithium_t=("recovered_lithium_t", "sum"))
    totals = grouped.groupby(["scenario", "year", "policy_scenario", "continent"], as_index=False).agg(
        total_recovered_lithium_t=("recovered_lithium_t", "sum")
    )
    grouped = grouped.merge(totals, on=["scenario", "year", "policy_scenario", "continent"], how="left")
    grouped["technology_share_pct"] = (
        grouped["recovered_lithium_t"] / grouped["total_recovered_lithium_t"] * 100.0
    ).fillna(0.0)
    return grouped


def build_share_delta():
    continent_map = country_continent_map()
    data = pd.concat(
        [read_routes(label, path, continent_map) for label, path in SCENARIOS],
        ignore_index=True,
    )
    years = sorted(data["year"].unique())
    index = pd.MultiIndex.from_product(
        [[label for label, _ in SCENARIOS], years, [BASELINE_POLICY] + COMPARISON_POLICIES, CONTINENTS, TECHNOLOGIES],
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
    delta["share_delta_vs_current_pp"] = (
        delta["technology_share_pct"] - delta["current_share_pct"].fillna(0.0)
    )
    return delta.groupby(["year", "policy_scenario", "continent", "technology"], as_index=False).agg(
        share_delta_vs_current_pp=("share_delta_vs_current_pp", "mean")
    )


def plot_global_pct_lines(ax, pct_data, policy, show_xlabel=True):
    """Line + shaded-band plot of % change vs current for all 3 technologies."""
    subset = pct_data[pct_data["policy_scenario"] == policy]
    years_all = sorted(subset["year"].unique())
    for tech in TECHNOLOGIES:
        color = TECHNOLOGY_COLORS[tech]
        td = subset[subset["technology"] == tech].sort_values("year")
        ax.plot(td["year"], td["pct_mean"], color=color, linewidth=1.8)
        ax.fill_between(td["year"], td["pct_min"], td["pct_max"], color=color, alpha=0.18)
    ax.axhline(0, color="#374151", linewidth=0.8, linestyle="--", alpha=0.55)
    ax.set_title(POLICY_LABELS[policy], fontsize=10.5, pad=6)
    ax.set_xlim(min(years_all), max(years_all))
    ax.set_xticks([2025, 2030, 2035, 2040, 2045, 2050])
    if show_xlabel:
        ax.set_xlabel("Year", fontsize=8.0)
    ax.set_ylabel("Change vs Current (%)", fontsize=8.0)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_linewidth(0.7)
        ax.spines[spine].set_color("#374151")
    ax.grid(axis="y", color="0.88", linewidth=0.5)


def plot_surface(ax, share_delta, technology, policy, show_ylabels=False):
    years = sorted(share_delta["year"].unique())
    x_grid, y_grid = np.meshgrid(years, range(len(CONTINENTS)))
    subset = share_delta[
        (share_delta["technology"] == technology)
        & (share_delta["policy_scenario"] == policy)
    ]
    surface = (
        subset.pivot_table(
            index="continent",
            columns="year",
            values="share_delta_vs_current_pp",
            aggfunc="first",
            fill_value=0.0,
        )
        .reindex(index=CONTINENTS, columns=years, fill_value=0.0)
    )
    ax.plot_surface(
        x_grid,
        y_grid,
        surface.values,
        color=POLICY_COLORS[policy],
        alpha=0.58,
        linewidth=0.25,
        edgecolor=POLICY_COLORS[policy],
        antialiased=True,
        shade=False,
    )
    ax.plot_wireframe(
        x_grid,
        y_grid,
        np.zeros_like(x_grid, dtype=float),
        color="#9CA3AF",
        linewidth=0.25,
        alpha=0.5,
    )
    ax.set_xlim(2025, 2050)
    ax.set_ylim(-0.3, len(CONTINENTS) - 0.7)
    ax.set_zlim(-45, 45)
    ax.set_xticks([2025, 2035, 2045, 2050])
    ax.set_yticks(range(len(CONTINENTS)))
    ax.set_yticklabels(CONTINENTS if show_ylabels else [], fontsize=6.3)
    ax.set_zticks([-40, 0, 40])
    ax.set_xlabel("Year", labelpad=3, fontsize=8)
    if show_ylabels:
        ax.set_ylabel("Continent", labelpad=4, fontsize=7.6)
    ax.set_zlabel("Share delta (pp)", labelpad=4, fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    ax.view_init(elev=24, azim=-58)
    ax.grid(True)


def plot_delta_heatmap(ax, share_delta, technology, policy, show_ylabels=False, norm=None):
    subset = share_delta[
        (share_delta["technology"] == technology)
        & (share_delta["policy_scenario"] == policy)
    ]
    years = sorted(subset["year"].unique())
    table = (
        subset.pivot_table(
            index="continent",
            columns="year",
            values="share_delta_vs_current_pp",
            aggfunc="first",
            fill_value=0.0,
        )
        .reindex(index=CONTINENTS, columns=years, fill_value=0.0)
    )
    image = ax.imshow(
        table.values,
        aspect="auto",
        cmap="RdBu_r",
        norm=norm,
        extent=[min(years) - 0.5, max(years) + 0.5, len(CONTINENTS) - 0.5, -0.5],
    )
    ax.axvline(2030, color="white", linewidth=0.4, alpha=0.45)
    ax.axvline(2040, color="white", linewidth=0.4, alpha=0.45)
    ax.axvline(2050, color="white", linewidth=0.4, alpha=0.45)
    ax.set_xlim(2025, 2050)
    ax.set_xticks([2025, 2030, 2035, 2040, 2045, 2050])
    ax.tick_params(axis="both", labelsize=7.6, length=0)
    ax.set_yticks(range(len(CONTINENTS)))
    ax.set_yticklabels(CONTINENTS if show_ylabels else [], fontsize=7.4)
    if show_ylabels:
        ax.set_ylabel(TECHNOLOGY_LABELS[technology], fontsize=10.0, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#374151")
    return image


TECHNOLOGY_COLORS = {
    "Direct": "#E15759",
    "Hydro": "#4E79A7",
    "PyroHydro": "#59A14F",
}


def plot_continent_radar_combined(ax, share_delta, policy, limit):
    """Draw all 3 technologies on one radar; dashed=2030, solid=2050."""
    angles = np.linspace(0, 2 * np.pi, len(CONTINENTS), endpoint=False).tolist()
    angles += angles[:1]
    for technology in TECHNOLOGIES:
        subset = share_delta[
            (share_delta["technology"] == technology)
            & (share_delta["policy_scenario"] == policy)
            & (share_delta["year"].isin([2030, 2050]))
        ].copy()
        color = TECHNOLOGY_COLORS[technology]
        for year, alpha, linewidth, linestyle in [
            (2030, 0.60, 1.3, "--"),
            (2050, 0.95, 2.0, "-"),
        ]:
            values = []
            for continent in CONTINENTS:
                row = subset[(subset["year"] == year) & (subset["continent"] == continent)]
                values.append(float(row["share_delta_vs_current_pp"].iloc[0]) if not row.empty else 0.0)
            scaled = [0.5 + 0.5 * v / limit for v in values]
            scaled += scaled[:1]
            ax.plot(angles, scaled, color=color, alpha=alpha, linewidth=linewidth, linestyle=linestyle)
            ax.fill(angles, scaled, color=color, alpha=0.06)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(CONTINENTS, fontsize=7.2)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.grid(color="0.84", linewidth=0.65)


def plot_combined(pct_data, share_delta):
    plt.rcParams.update({"font.family": "Arial"})
    # Layout: 3 rows × 2 cols; left col = line plots, right col = radars
    fig = plt.figure(figsize=(11.2, 11.4), dpi=300)
    outer = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.28,
                             left=0.09, right=0.97, bottom=0.14, top=0.93)

    radar_limit = 8.0

    for row_idx, policy in enumerate(COMPARISON_POLICIES):
        # --- left: pct-change line + band ---
        ax_line = fig.add_subplot(outer[row_idx, 0])
        show_x = row_idx == 2
        plot_global_pct_lines(ax_line, pct_data, policy, show_xlabel=show_x)
        if row_idx != 2:
            ax_line.tick_params(axis="x", labelbottom=False)

        # --- right: combined radar ---
        ax_radar = fig.add_subplot(outer[row_idx, 1], projection="polar")
        plot_continent_radar_combined(ax_radar, share_delta, policy, radar_limit)
        ax_radar.set_title(POLICY_LABELS[policy], fontsize=9.5, pad=12)

    from matplotlib.patches import Patch
    # Shared legend: technology colors (used in both columns) + year linestyles (radar) + scale note
    tech_handles = [
        Line2D([0], [0], color=TECHNOLOGY_COLORS[t], lw=2.0, label=TECHNOLOGY_LABELS[t])
        for t in TECHNOLOGIES
    ]
    year_handles = [
        Line2D([0], [0], color="#555555", lw=1.4, linestyle="--", label="Radar 2030"),
        Line2D([0], [0], color="#555555", lw=2.0, linestyle="-",  label="Radar 2050"),
    ]
    scale_handle = Patch(
        facecolor="none", edgecolor="none",
        label=f"Radar scale: outer = +{radar_limit:.0f} pp,  center = 0,  inner = −{radar_limit:.0f} pp",
    )
    fig.legend(
        handles=tech_handles + year_handles + [scale_handle],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.02),
        fontsize=8.8,
    )

    fig.suptitle(
        "Policy impacts on technology selection relative to Current",
        fontsize=12.5,
        y=0.97,
    )
    fig.text(0.27, 0.005, "Left: annual % change in recovered Li vs Current (shading = scenario range)",
             ha="center", fontsize=7.8, color="#374151")
    fig.text(0.73, 0.005, "Right: continent-level technology-share delta (pp), dashed=2030, solid=2050",
             ha="center", fontsize=7.8, color="#374151")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "policy_impact_radar_technology_surface.png"
    pdf = OUT_DIR / "policy_impact_radar_technology_surface.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


def main():
    pct_data = build_global_pct_change()
    share_delta = build_share_delta()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pct_data.to_csv(OUT_DIR / "policy_impact_global_pct_change.csv", index=False)
    share_delta.to_csv(OUT_DIR / "policy_impact_technology_continent_share_delta.csv", index=False)
    plot_combined(pct_data, share_delta)


if __name__ == "__main__":
    main()
