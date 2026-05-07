from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "Figure_data"
BARRIER_FILE = (
    ROOT
    / "trans"
    / "scenario_result"
    / "high_collection"
    / "baseline"
    / "barrier_decomposition"
    / "lithium_barrier_decomposition.csv"
)
SCRAP_TOTAL_FILE = ROOT / "Scenario result" / "EV_battery_inuse_and_manufacturing_scrap_total.csv"
CAPACITY_FILE = ROOT / "recycling_cap_2050.csv"
POSITIONS_FILE = ROOT / "nation_lat_lon.csv"
ROUTE_LOSS_FILE = (
    ROOT
    / "Figure_data"
    / "joint_policy_technology"
    / "route_access_loss_by_route.csv"
)
LOSS_COMPARISON_FILE = OUTPUT_DIR / "lithium_loss_comparison.csv"

YEARS = [2025, 2030, 2035, 2040, 2045, 2050]
SELECTED_STRATEGY = "Strategy 3"
SELECTED_MODE = "Realistic_multiobjective"
MAP_YEAR = 2050
ROUTE_POLICY = "strict_policy"
CAPTURE_SCENARIO = "high_collection"
REGION_ORDER = ["Asia", "Europe", "America", "Africa", "Oceania", "Other"]
REGION_COLORS = {
    "Asia": "#2A9D8F",
    "Europe": "#4E79A7",
    "America": "#E15759",
    "Africa": "#F28E2B",
    "Oceania": "#B07AA1",
    "Other": "#9CA3AF",
}
COUNTRY_COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#6B7280",
]
EU_COUNTRIES = {
    "Austria",
    "Belgium",
    "Bulgaria",
    "Croatia",
    "Cyprus",
    "Czech Republic",
    "Denmark",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Ireland",
    "Italy",
    "Latvia",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Netherlands",
    "Poland",
    "Portugal",
    "Romania",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Sweden",
}


COUNTRY_ALIASES = {
    "USA": "United States",
    "United States of America": "United States",
    "Korea": "Republic of Korea",
    "South Korea": "Republic of Korea",
    "Russian Federation": "Russia",
    "Viet Nam": "Vietnam",
    "Czechia": "Czech Republic",
    "Turkey": "Turkiye",
}


def clean_country(value):
    text = str(value).strip()
    return COUNTRY_ALIASES.get(text, text)


def load_barrier():
    data = pd.read_csv(BARRIER_FILE)
    data = data[
        (data["Strategy type"] == SELECTED_STRATEGY)
        & (data["choice_mode"] == SELECTED_MODE)
        & (data["year"].isin(YEARS))
    ].copy()
    data = data.sort_values("year")
    keep = [
        "year",
        "embedded_li",
        "collection_loss",
        "technology_loss",
        "trade_policy_loss",
        "economic_selection_loss",
        "supply_chain_available_secondary_li",
    ]
    data = data[keep].copy()
    for col in keep:
        if col != "year":
            data[col + "_kt"] = data[col] / 1000.0
    return data


def load_positions():
    positions = pd.read_csv(POSITIONS_FILE)
    positions["country_clean"] = positions["country"].map(clean_country)
    positions["region_group"] = positions["continent"].where(
        positions["continent"].isin(REGION_ORDER), "Other"
    )
    return positions


def load_country_scrap():
    scrap = pd.read_csv(SCRAP_TOTAL_FILE)
    scrap = scrap[scrap["Year"] == MAP_YEAR].copy()
    scrap["country_clean"] = scrap["region"].map(clean_country)
    grouped = scrap.groupby("country_clean", as_index=False)["scrap"].sum()
    return grouped.rename(columns={"scrap": "scrap_t"})


def load_capacity():
    cap = pd.read_csv(CAPACITY_FILE)
    cap = cap[cap["Year"] == MAP_YEAR].copy()
    long = cap.melt(id_vars=["Year"], var_name="country", value_name="capacity")
    long["country_clean"] = long["country"].map(clean_country)
    long["capacity"] = pd.to_numeric(long["capacity"], errors="coerce").fillna(0.0)
    return long.groupby("country_clean", as_index=False)["capacity"].sum()


def load_map_points():
    positions = load_positions()
    scrap = load_country_scrap()
    cap = load_capacity()
    points = positions.merge(scrap, on="country_clean", how="left").merge(
        cap, on="country_clean", how="left"
    )
    points[["scrap_t", "capacity"]] = points[["scrap_t", "capacity"]].fillna(0.0)
    return points


def load_routes():
    routes = pd.read_csv(ROUTE_LOSS_FILE)
    routes = routes[
        (routes["year"] == MAP_YEAR) & (routes["policy_scenario"] == ROUTE_POLICY)
    ].copy()
    routes["route_access_loss_li_t"] = pd.to_numeric(
        routes["route_access_loss_li_t"], errors="coerce"
    ).fillna(0.0)
    routes = routes.sort_values("route_access_loss_li_t", ascending=False).head(14)
    return routes


def load_source_region_cascade():
    positions = load_positions()
    by_country = positions[["iso3", "country_clean", "region_group"]].drop_duplicates()
    by_iso = positions[["iso3", "region_group"]].drop_duplicates()

    capture = pd.read_csv(OUTPUT_DIR / "capture_loss_with_manufacturing_country.csv")
    capture = capture[
        (capture["Year"] == MAP_YEAR) & (capture["scenario"] == CAPTURE_SCENARIO)
    ].copy()
    capture["country_clean"] = capture["country"].map(clean_country)
    capture = capture.merge(by_country, on="country_clean", how="left")
    capture["region_group"] = capture["region_group"].fillna("Other")
    capture_region = capture.groupby("region_group", as_index=True)[
        ["total_available_lithium_t", "total_captured_lithium_t"]
    ].sum()

    routes = pd.read_csv(
        OUTPUT_DIR
        / "joint_policy_technology"
        / "lithium_loss_scenarios"
        / "lithium_loss_scenarios_routes.csv"
    )
    routes = routes[
        (routes["year"] == MAP_YEAR)
        & (routes["policy_scenario"] == "current_policy")
        & (routes["mitigation_scenario"] == "baseline")
        & (routes["strategy"] == SELECTED_STRATEGY)
        & (~routes["is_unprocessed"].astype(bool))
    ].copy()
    routes = routes.merge(by_iso, left_on="source_iso3", right_on="iso3", how="left")
    routes["region_group"] = routes["region_group"].fillna("Other")
    recovered_region = routes.groupby("region_group", as_index=True)[
        "recovered_lithium_t"
    ].sum()

    policy_routes = pd.read_csv(ROUTE_LOSS_FILE)
    policy_routes = policy_routes[
        (policy_routes["year"] == MAP_YEAR)
        & (policy_routes["policy_scenario"] == ROUTE_POLICY)
    ].copy()
    policy_routes = policy_routes.merge(
        by_iso, left_on="source_iso3", right_on="iso3", how="left"
    )
    policy_routes["region_group"] = policy_routes["region_group"].fillna("Other")
    policy_region = policy_routes.groupby("region_group", as_index=True)[
        "route_access_loss_li_t"
    ].sum()

    rows = []
    for region in REGION_ORDER:
        potential = float(
            capture_region.loc[region, "total_available_lithium_t"]
            if region in capture_region.index
            else 0.0
        )
        captured = float(
            capture_region.loc[region, "total_captured_lithium_t"]
            if region in capture_region.index
            else 0.0
        )
        available = float(
            recovered_region.loc[region] if region in recovered_region.index else 0.0
        )
        policy_loss = float(
            policy_region.loc[region] if region in policy_region.index else 0.0
        )
        policy_accessible = max(available, available + policy_loss * 0.0)
        technically_recoverable = max(policy_accessible, available)
        rows.append(
            {
                "region_group": region,
                "Potential secondary Li": potential / 1000.0,
                "After capture": captured / 1000.0,
                "Supply-chain available": available / 1000.0,
                "Policy-route disruption": policy_loss / 1000.0,
            }
        )
    data = pd.DataFrame(rows)
    return data


def load_top_country_cascade(top_n=8):
    positions = load_positions()
    by_country = positions[["iso3", "country_clean"]].drop_duplicates()
    iso_to_country = by_country.set_index("iso3")["country_clean"].to_dict()

    capture = pd.read_csv(OUTPUT_DIR / "capture_loss_with_manufacturing_country.csv")
    capture = capture[
        (capture["Year"] == MAP_YEAR) & (capture["scenario"] == CAPTURE_SCENARIO)
    ].copy()
    capture["country_clean"] = capture["country"].map(clean_country)
    capture = capture.groupby("country_clean", as_index=True)[
        ["total_available_lithium_t", "total_captured_lithium_t"]
    ].sum()

    routes = pd.read_csv(
        OUTPUT_DIR
        / "joint_policy_technology"
        / "lithium_loss_scenarios"
        / "lithium_loss_scenarios_routes.csv"
    )
    routes = routes[
        (routes["year"] == MAP_YEAR)
        & (routes["mitigation_scenario"] == "baseline")
        & (routes["strategy"] == SELECTED_STRATEGY)
        & (~routes["is_unprocessed"].astype(bool))
        & (routes["policy_scenario"].isin(["route_access_open", ROUTE_POLICY]))
    ].copy()
    routes["country_clean"] = routes["source_iso3"].map(iso_to_country).fillna("Other")
    recovered = routes.pivot_table(
        index="country_clean",
        columns="policy_scenario",
        values="recovered_lithium_t",
        aggfunc="sum",
        fill_value=0.0,
    )
    for column in ["route_access_open", ROUTE_POLICY]:
        if column not in recovered:
            recovered[column] = 0.0

    top_countries = (
        capture["total_available_lithium_t"].sort_values(ascending=False).head(top_n).index
    )
    rows = []
    for country in list(top_countries) + ["Other"]:
        if country == "Other":
            capture_rows = capture[~capture.index.isin(top_countries)]
            recovered_rows = recovered[~recovered.index.isin(top_countries)]
            potential = float(capture_rows["total_available_lithium_t"].sum())
            captured = float(capture_rows["total_captured_lithium_t"].sum())
            open_recovered = float(recovered_rows["route_access_open"].sum())
            policy_recovered = float(recovered_rows[ROUTE_POLICY].sum())
        else:
            potential = float(
                capture.loc[country, "total_available_lithium_t"]
                if country in capture.index
                else 0.0
            )
            captured = float(
                capture.loc[country, "total_captured_lithium_t"]
                if country in capture.index
                else 0.0
            )
            open_recovered = float(
                recovered.loc[country, "route_access_open"]
                if country in recovered.index
                else 0.0
            )
            policy_recovered = float(
                recovered.loc[country, ROUTE_POLICY]
                if country in recovered.index
                else 0.0
            )
        rows.append(
            {
                "country": country,
                "Potential secondary Li": potential / 1000.0,
                "After capture": captured / 1000.0,
                "Open-access recovered": open_recovered / 1000.0,
                "Policy-accessible recovered": policy_recovered / 1000.0,
            }
        )
    return pd.DataFrame(rows)


def load_source_to_destination_cascade(top_n=5, year=MAP_YEAR):
    positions = load_positions()
    by_country = positions[["iso3", "country_clean"]].drop_duplicates()
    iso_to_country = by_country.set_index("iso3")["country_clean"].to_dict()

    capture = pd.read_csv(OUTPUT_DIR / "capture_loss_with_manufacturing_country.csv")
    capture = capture[
        (capture["Year"] == year) & (capture["scenario"] == CAPTURE_SCENARIO)
    ].copy()
    capture["country_clean"] = capture["country"].map(clean_country)
    capture = capture.groupby("country_clean", as_index=True)[
        ["total_available_lithium_t", "total_captured_lithium_t"]
    ].sum()

    routes = pd.read_csv(
        OUTPUT_DIR
        / "joint_policy_technology"
        / "lithium_loss_scenarios"
        / "lithium_loss_scenarios_routes.csv"
    )
    routes = routes[
        (routes["year"] == year)
        & (routes["mitigation_scenario"] == "baseline")
        & (routes["strategy"] == SELECTED_STRATEGY)
        & (~routes["is_unprocessed"].astype(bool))
        & (routes["policy_scenario"].isin(["route_access_open", ROUTE_POLICY]))
    ].copy()
    routes["destination_country"] = routes["destination_iso3"].map(iso_to_country).fillna("Other")
    dest_recovered = routes.pivot_table(
        index="destination_country",
        columns="policy_scenario",
        values="recovered_lithium_t",
        aggfunc="sum",
        fill_value=0.0,
    )
    for column in ["route_access_open", ROUTE_POLICY]:
        if column not in dest_recovered:
            dest_recovered[column] = 0.0

    priority = [
        "India",
        "China",
        "United States",
        "Republic of Korea",
        "European Union",
        "Other",
    ]
    selected = set(label for label in priority if label not in {"Other", "European Union"})
    labels = priority

    rows = []
    for label in labels:
        if label == "Other":
            excluded = selected | EU_COUNTRIES
            source_rows = capture[~capture.index.isin(excluded)]
            dest_rows = dest_recovered[~dest_recovered.index.isin(excluded)]
            potential = float(source_rows["total_available_lithium_t"].sum())
            captured = float(source_rows["total_captured_lithium_t"].sum())
            open_recovered = float(dest_rows["route_access_open"].sum())
            policy_recovered = float(dest_rows[ROUTE_POLICY].sum())
        elif label == "European Union":
            source_rows = capture[capture.index.isin(EU_COUNTRIES)]
            dest_rows = dest_recovered[dest_recovered.index.isin(EU_COUNTRIES)]
            potential = float(source_rows["total_available_lithium_t"].sum())
            captured = float(source_rows["total_captured_lithium_t"].sum())
            open_recovered = float(dest_rows["route_access_open"].sum())
            policy_recovered = float(dest_rows[ROUTE_POLICY].sum())
        else:
            potential = float(
                capture.loc[label, "total_available_lithium_t"]
                if label in capture.index
                else 0.0
            )
            captured = float(
                capture.loc[label, "total_captured_lithium_t"]
                if label in capture.index
                else 0.0
            )
            open_recovered = float(
                dest_recovered.loc[label, "route_access_open"]
                if label in dest_recovered.index
                else 0.0
            )
            policy_recovered = float(
                dest_recovered.loc[label, ROUTE_POLICY]
                if label in dest_recovered.index
                else 0.0
            )
        rows.append(
            {
                "country": label,
                "Potential by source": potential / 1000.0,
                "Captured by source": captured / 1000.0,
                "Open-access by destination": open_recovered / 1000.0,
                f"{ROUTE_POLICY} by destination": policy_recovered / 1000.0,
            }
        )
    return pd.DataFrame(rows)


def load_loss_comparison():
    data = pd.read_csv(LOSS_COMPARISON_FILE)
    data = data[data["year"].isin(YEARS)].copy()
    return data.sort_values("year")


def size_from(values, min_size, max_size):
    values = np.asarray(values, dtype=float)
    if values.max() <= 0:
        return np.full(len(values), min_size)
    scaled = np.sqrt(values / values.max())
    return min_size + scaled * (max_size - min_size)


def draw_world_background(ax):
    ax.set_xlim(-180, 180)
    ax.set_ylim(-58, 78)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#F8FAFC")
    ax.axhline(0, color="#E5E7EB", linewidth=0.6, zorder=0)
    for lon in range(-120, 181, 60):
        ax.axvline(lon, color="#EEF2F7", linewidth=0.5, zorder=0)
    for lat in [-40, 0, 40]:
        ax.axhline(lat, color="#EEF2F7", linewidth=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_a(ax, barrier):
    years = barrier["year"].to_numpy()
    embedded = barrier["embedded_li_kt"].to_numpy()
    available = barrier["supply_chain_available_secondary_li_kt"].to_numpy()
    lost = embedded - available
    ax.fill_between(years, embedded, color="#CBD5E1", alpha=0.55, label="Embedded Li potential")
    ax.fill_between(years, available, color="#10B981", alpha=0.72, label="Supply-chain available")
    ax.plot(years, embedded, color="#64748B", linewidth=2.0)
    ax.plot(years, available, color="#047857", linewidth=2.2)
    ax.fill_between(years, available, embedded, color="#EF4444", alpha=0.15, label="Barrier gap")
    ax.annotate(
        f"{embedded[-1]:,.0f} kt Li potential",
        xy=(years[-1], embedded[-1]),
        xytext=(-92, -22),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": "#475569", "linewidth": 0.8},
        fontsize=8.5,
        color="#334155",
    )
    ax.annotate(
        f"{lost[-1]:,.0f} kt Li gap",
        xy=(years[-1], (embedded[-1] + available[-1]) / 2),
        xytext=(-82, 24),
        textcoords="offset points",
        arrowprops={"arrowstyle": "-", "color": "#B91C1C", "linewidth": 0.8},
        fontsize=8.5,
        color="#991B1B",
    )
    ax.set_title("A. A rising secondary-lithium opportunity", loc="left", fontweight="bold")
    ax.set_ylabel("Lithium (kt Li)")
    ax.set_xticks(years)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def panel_b(ax, points):
    draw_world_background(ax)
    generation = points[points["scrap_t"] > 0].copy()
    capacity = points[points["capacity"] > 0].copy()
    ax.scatter(
        generation["lon"],
        generation["lat"],
        s=size_from(generation["scrap_t"], 8, 260),
        color="#2563EB",
        alpha=0.34,
        edgecolor="none",
        label="Secondary Li generation proxy",
        zorder=2,
    )
    ax.scatter(
        capacity["lon"],
        capacity["lat"],
        s=size_from(capacity["capacity"], 22, 260),
        facecolors="none",
        edgecolors="#F97316",
        linewidths=1.4,
        alpha=0.86,
        label="Recycling capacity",
        zorder=3,
    )
    top = generation.sort_values("scrap_t", ascending=False).head(6)
    for _, row in top.iterrows():
        ax.text(row["lon"] + 3, row["lat"] + 2, row["iso3"], fontsize=7.5, color="#1E3A8A")
    ax.set_title("B. Potential and processing capacity are spatially mismatched", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower left")


def panel_c(ax, points, routes):
    draw_world_background(ax)
    positions = points.set_index("iso3")
    values = routes["route_access_loss_li_t"].to_numpy()
    for _, row in routes.iterrows():
        src = row["source_iso3"]
        dst = row["destination_iso3"]
        if src not in positions.index or dst not in positions.index:
            continue
        src_row = positions.loc[src]
        dst_row = positions.loc[dst]
        value = float(row["route_access_loss_li_t"])
        width = 0.7 + 4.2 * np.sqrt(value / max(values.max(), 1.0))
        ax.plot(
            [src_row["lon"], dst_row["lon"]],
            [src_row["lat"], dst_row["lat"]],
            color="#DC2626",
            linewidth=width,
            alpha=0.48,
            solid_capstyle="round",
            zorder=2,
        )
        ax.scatter([src_row["lon"], dst_row["lon"]], [src_row["lat"], dst_row["lat"]], s=16, color="#111827", zorder=3)
    ax.text(
        0.02,
        0.04,
        f"{ROUTE_POLICY.replace('_', ' ')}; line width = affected route Li",
        transform=ax.transAxes,
        fontsize=8,
        color="#7F1D1D",
    )
    ax.set_title("C. Trade policy reshapes route accessibility", loc="left", fontweight="bold")


def availability_path(loss_row, barrier):
    potential = float(
        barrier.loc[barrier["year"] == int(loss_row["year"]), "embedded_li_kt"].iloc[0]
    )
    captured = potential - loss_row["capture_loss_lithium_kt"]
    technically_recoverable = captured - loss_row["technology_recovery_loss_kt"]
    policy_accessible = technically_recoverable - loss_row["trade_policy_loss_kt"]
    supply_chain_available = policy_accessible - loss_row["economic_selection_loss_kt"]
    return np.array(
        [
            potential,
            captured,
            technically_recoverable,
            policy_accessible,
            supply_chain_available,
        ],
        dtype=float,
    )


def panel_d(ax, losses, barrier):
    cascade = load_source_to_destination_cascade(year=MAP_YEAR)
    stage_cols = [
        "Potential by source",
        "Captured by source",
        "Open-access by destination",
        f"{ROUTE_POLICY} by destination",
    ]
    stage_labels = [
        "Potential\nsource",
        "Captured\nsource",
        "Open access\ndestination",
        f"{ROUTE_POLICY.replace('_', ' ').title()}\ndestination",
    ]
    x = np.arange(len(stage_labels))
    bottom = np.zeros(len(stage_cols))
    for idx, row in cascade.iterrows():
        country = row["country"]
        if row.empty:
            continue
        values = row[stage_cols].to_numpy(dtype=float)
        ax.fill_between(
            x,
            bottom,
            bottom + values,
            color=COUNTRY_COLORS[min(idx, len(COUNTRY_COLORS) - 1)],
            alpha=0.78,
            linewidth=0,
            label=country,
            zorder=1,
        )
        ax.plot(
            x,
            bottom + values,
            color="white",
            linewidth=0.7,
            alpha=0.8,
            zorder=2,
        )
        bottom += values
    totals = cascade[stage_cols].sum().to_numpy(dtype=float)
    marker_colors = ["#64748B", "#2563EB", "#F59E0B", "#2A9D8F"]
    ax.plot(x, totals, color="#111827", linewidth=1.5, zorder=3)
    ax.scatter(x, totals, s=78, color=marker_colors, edgecolor="white", linewidth=1.0, zorder=4)
    for i, value in enumerate(totals):
        ax.text(i, value + totals.max() * 0.025, f"{value:,.0f}", ha="center", va="bottom", fontsize=8)
    barrier_labels = [
        ("capture / collection", 0.5),
        ("technology + routing", 1.5),
        ("policy access", 2.5),
    ]
    y_label = totals.max() * 0.055
    for label, xpos in barrier_labels:
        ax.text(
            xpos,
            y_label,
            label,
            ha="center",
            va="bottom",
            fontsize=7.4,
            color="#475569",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, fontsize=7.3)
    ax.set_ylabel("Lithium (kt Li)")
    ax.set_title(f"D. Source potential to destination access in {MAP_YEAR}", loc="left", fontweight="bold", pad=12)
    ax.set_ylim(0, totals.max() * 1.22)
    ax.set_xlim(-0.25, len(stage_labels) - 0.05)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        frameon=False,
        fontsize=7.2,
        loc="upper right",
        ncol=2,
        title="Key countries",
        title_fontsize=7.5,
    )
    open_total = totals[2]
    policy_total = totals[3]
    ax.annotate(
        f"Policy access reshapes destination control\n{open_total - policy_total:,.0f} kt Li-equivalent gap under strict policy",
        xy=(2.85, policy_total),
        xytext=(1.72, totals.max() * 0.68),
        arrowprops={"arrowstyle": "->", "color": "#334155", "linewidth": 0.9},
        fontsize=8.0,
        color="#334155",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#CBD5E1", "alpha": 0.9},
    )


def write_panel_data(barrier, points, routes):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    barrier.to_csv(OUTPUT_DIR / "Figure1_concept_barrier_data.csv", index=False)
    points.to_csv(OUTPUT_DIR / "Figure1_concept_map_points.csv", index=False)
    routes.to_csv(OUTPUT_DIR / "Figure1_concept_policy_routes.csv", index=False)
    load_source_region_cascade().to_csv(
        OUTPUT_DIR / "Figure1_concept_region_cascade.csv", index=False
    )
    load_top_country_cascade().to_csv(
        OUTPUT_DIR / "Figure1_concept_top_country_cascade.csv", index=False
    )
    load_source_to_destination_cascade().to_csv(
        OUTPUT_DIR / "Figure1_concept_source_destination_cascade.csv", index=False
    )
    pd.concat(
        [
            load_source_to_destination_cascade(year=year).assign(year=year)
            for year in [2030, 2040, 2050]
        ],
        ignore_index=True,
    ).to_csv(
        OUTPUT_DIR / "Figure1_concept_source_destination_cascade_2030_2040_2050.csv",
        index=False,
    )


def main():
    barrier = load_barrier()
    losses = load_loss_comparison()
    points = load_map_points()
    routes = load_routes()
    write_panel_data(barrier, points, routes)

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(15.6, 9.4), dpi=220)
    grid = fig.add_gridspec(2, 2, width_ratios=[0.92, 1.34], height_ratios=[0.92, 1.08], hspace=0.34, wspace=0.26)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    panel_a(ax_a, barrier)
    panel_b(ax_b, points)
    panel_c(ax_c, points, routes)
    panel_d(ax_d, losses, barrier)

    fig.suptitle(
        "Global secondary lithium potential is rising, but battery-supply-chain access is constrained",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.947,
        "Collection, technology, economic choice and trade-policy accessibility barriers reduce battery-supply-chain-available secondary lithium.",
        ha="center",
        fontsize=9.5,
        color="#475569",
    )
    output = OUTPUT_DIR / "Figure1_concept_supply_chain_access.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")
    print(f"Wrote {OUTPUT_DIR / 'Figure1_concept_barrier_data.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'Figure1_concept_map_points.csv'}")
    print(f"Wrote {OUTPUT_DIR / 'Figure1_concept_policy_routes.csv'}")


if __name__ == "__main__":
    main()
