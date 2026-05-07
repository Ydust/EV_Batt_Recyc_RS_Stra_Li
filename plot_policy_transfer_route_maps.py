import argparse
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap


ROOT = Path(__file__).resolve().parent
ROUTES_FILE = (
    ROOT
    / "Figure_data"
    / "policy_constraint_strength"
    / "policy_constraint_strength_used_routes.csv"
)
COUNTRY_FILE = ROOT / "all_countries.csv"
OUTPUT_DIR = ROOT / "Figure_data" / "policy_constraint_strength"

POLICIES = [
    "current_policy",
    "reference_policy",
    "strict_policy",
    "critical_route_policy",
    "domestic_processing_policy",
]

POLICY_TITLES = {
    "route_access_open": "Route-access-unconstrained benchmark",
    "current_policy": "Current policy",
    "reference_policy": "Reference policy vs current",
    "strict_policy": "Strict policy vs current",
    "critical_route_policy": "Critical-route policy vs current",
    "domestic_processing_policy": "Domestic processing vs current",
}

CONTINENT_PIE_COLORS = {
    "Africa": "#ff9999",
    "Americas": "#c2c2f0",
    "Asia": "#66b3ff",
    "Europe": "#99ff99",
    "Oceania": "#ffcc99",
    "Other": "#D1D5DB",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot four policy-scenario transfer-route maps. The current-policy "
            "panel shows absolute flows; the other panels show route deltas "
            "relative to current_policy."
        )
    )
    parser.add_argument("--year", type=int, default=2030)
    parser.add_argument("--method", default="Direct")
    parser.add_argument("--strategy", default="Strategy 3")
    parser.add_argument("--top-routes", type=int, default=70)
    parser.add_argument("--min-delta", type=float, default=1e-9)
    parser.add_argument(
        "--flow-column",
        default="scrap_t",
        help="Route-flow column used for line widths and continent pies.",
    )
    parser.add_argument(
        "--routes-file",
        default=str(ROUTES_FILE),
        help="Route table to plot. Defaults to the policy-constraint route table.",
    )
    parser.add_argument(
        "--policies",
        default=",".join(POLICIES),
        help="Comma-separated policy scenarios to plot, in panel order.",
    )
    parser.add_argument(
        "--baseline-policy",
        default="current_policy",
        help="Policy scenario used as the absolute-flow and delta baseline panel.",
    )
    parser.add_argument(
        "--route-access-reference-policy",
        default="",
        help=(
            "Policy scenario used to calculate route access loss. "
            "If omitted, the plotting baseline policy is used."
        ),
    )
    parser.add_argument(
        "--route-access-loss-column",
        default="battery_embedded_secondary_li_t",
        help="Lithium column used to calculate route access loss.",
    )
    parser.add_argument(
        "--joint-routes",
        action="store_true",
        help="Read the joint policy-transport-technology route output.",
    )
    parser.add_argument(
        "--show-continent-pies",
        action="store_true",
        help="Add top-node continent-composition pie charts to each panel.",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_DIR / "policy_transfer_route_maps_2030_Direct_Strategy3.png"),
    )
    return parser.parse_args()


def load_positions():
    countries = pd.read_csv(COUNTRY_FILE)
    countries = countries.dropna(subset=["iso3", "lat", "lon"]).copy()
    return countries.set_index("iso3")[["country", "lat", "lon", "continent"]]


def parse_policies(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def load_routes(year, method, strategy, routes_file, policies, joint_routes):
    routes = pd.read_csv(routes_file)
    filters = (
        (routes["year"] == year)
        & (routes["strategy"] == strategy)
        & (~routes["source_iso3"].astype(str).str.startswith("Virtual"))
        & (~routes["destination_iso3"].astype(str).str.startswith("Virtual"))
        & (routes["policy_scenario"].isin(policies))
    )
    if joint_routes:
        filters = filters & (~routes["is_unprocessed"].astype(bool))
        if method.lower() != "all":
            filters = filters & (routes["technology"] == method)
    else:
        filters = (
            filters
            & (routes["method"] == method)
            & (routes["cross_border"] == True)
        )
    scenario_rows = routes[filters].copy()
    scenario_rows = scenario_rows[
        ~scenario_rows["source_iso3"].astype(str).str.startswith("Virtual")
        & ~scenario_rows["destination_iso3"].astype(str).str.startswith("Virtual")
    ].copy()
    missing = set(policies) - set(scenario_rows["policy_scenario"].unique())
    routes = scenario_rows[scenario_rows["source_iso3"] != scenario_rows["destination_iso3"]].copy()
    if routes.empty:
        raise ValueError(
            f"No route rows found for year={year}, method={method}, strategy={strategy}."
        )
    if missing:
        raise ValueError(f"Missing policy scenarios in route table: {sorted(missing)}")
    return routes


def aggregate_policy_routes(routes, policy, value_col="scrap_t"):
    subset = routes[routes["policy_scenario"] == policy].copy()
    if value_col not in subset.columns:
        raise ValueError(f"Route flow column not found: {value_col}")
    return (
        subset.groupby(["source_iso3", "destination_iso3"], as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: policy})
    )


def aggregate_policy_li(routes, policy, value_col):
    subset = routes[routes["policy_scenario"] == policy].copy()
    if value_col not in subset.columns:
        raise ValueError(f"Route access loss column not found: {value_col}")
    return (
        subset.groupby(["source_iso3", "destination_iso3"], as_index=False)[value_col]
        .sum()
        .rename(columns={value_col: policy})
    )


def build_delta_table(routes, policies, baseline_policy, value_col, min_delta):
    current = aggregate_policy_routes(routes, baseline_policy, value_col)
    frames = []
    for policy in [item for item in policies if item != baseline_policy]:
        comparison = aggregate_policy_routes(routes, policy, value_col)
        merged = current.merge(
            comparison, on=["source_iso3", "destination_iso3"], how="outer"
        ).fillna(0)
        merged["policy_scenario"] = policy
        merged["delta_route_value"] = merged[policy] - merged[baseline_policy]
        merged = merged[merged["delta_route_value"].abs() > min_delta].copy()
        frames.append(
            merged[
                [
                    "policy_scenario",
                    "source_iso3",
                    "destination_iso3",
                    baseline_policy,
                    policy,
                    "delta_route_value",
                ]
            ].rename(columns={baseline_policy: "baseline_policy", policy: "comparison_policy"})
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_access_loss_table(routes, policies, reference_policy, value_col, min_delta):
    reference = aggregate_policy_li(routes, reference_policy, value_col)
    frames = []
    for policy in policies:
        comparison = aggregate_policy_li(routes, policy, value_col)
        merged = reference.merge(
            comparison, on=["source_iso3", "destination_iso3"], how="outer"
        ).fillna(0)
        merged["policy_scenario"] = policy
        merged["access_delta_li_t"] = merged[policy] - merged[reference_policy]
        merged = merged[merged["access_delta_li_t"].abs() > min_delta].copy()
        frames.append(
            merged[
                [
                    "policy_scenario",
                    "source_iso3",
                    "destination_iso3",
                    reference_policy,
                    policy,
                    "access_delta_li_t",
                ]
            ].rename(
                columns={
                    reference_policy: "route_access_reference_policy",
                    policy: "policy_li_t",
                }
            )
        )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def make_map(ax):
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
    return m


def route_width(values, value):
    vmax = max(float(values.max()), 1.0)
    return 0.45 + 3.2 * (np.log10(abs(value) + 1) / np.log10(vmax + 1))


def top_route_node_values(route_table, positions, mode, top_routes):
    if route_table.empty:
        return {}
    value_col = "route_value" if mode == "absolute" else "delta_route_value"
    table = route_table.copy()
    table["abs_value"] = table[value_col].abs()
    table = table.sort_values("abs_value", ascending=False).head(top_routes)
    node_values = {}
    for _, row in table.iterrows():
        src = row["source_iso3"]
        dst = row["destination_iso3"]
        if src not in positions.index or dst not in positions.index:
            continue
        node_values[src] = node_values.get(src, 0.0) + row["abs_value"]
        node_values[dst] = node_values.get(dst, 0.0) + row["abs_value"]
    return node_values


def draw_routes(ax, m, route_table, positions, mode, top_routes, node_scale_max=None):
    if route_table.empty:
        if mode == "delta":
            ax.text(
                0.5,
                0.5,
                "No route changes\nvs current policy",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                color="#4B5563",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=4),
                zorder=10,
            )
        return

    value_col = "route_value" if mode == "absolute" else "delta_route_value"
    route_table = route_table.copy()
    route_table["abs_value"] = route_table[value_col].abs()
    route_table = route_table.sort_values("abs_value", ascending=False).head(top_routes)
    values = route_table["abs_value"]

    node_values = {}
    for _, row in route_table.iterrows():
        src = row["source_iso3"]
        dst = row["destination_iso3"]
        if src not in positions.index or dst not in positions.index:
            continue
        src_lon, src_lat = positions.loc[src, ["lon", "lat"]]
        dst_lon, dst_lat = positions.loc[dst, ["lon", "lat"]]
        x1, y1 = m(src_lon, src_lat)
        x2, y2 = m(dst_lon, dst_lat)

        if mode == "absolute":
            color = "#2563EB"
            alpha = 0.48
            rad = 0.18
        elif row[value_col] > 0:
            color = "#DC2626"
            alpha = 0.62
            rad = 0.18
        else:
            color = "#2563EB"
            alpha = 0.55
            rad = -0.18

        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="->",
                connectionstyle=f"arc3,rad={rad}",
                linewidth=route_width(values, row["abs_value"]),
                color=color,
                alpha=alpha,
                shrinkA=1,
                shrinkB=1,
            ),
            zorder=4,
        )
        node_values[src] = node_values.get(src, 0.0) + row["abs_value"]
        node_values[dst] = node_values.get(dst, 0.0) + row["abs_value"]

    if not node_values:
        return

    max_node = node_scale_max or max(node_values.values())
    for iso3, value in node_values.items():
        lon, lat = positions.loc[iso3, ["lon", "lat"]]
        x, y = m(lon, lat)
        size = 12 + 120 * np.sqrt(min(value / max_node, 1.0))
        ax.scatter(
            x,
            y,
            s=size,
            facecolors="white",
            edgecolors="#111827",
            linewidths=0.5,
            alpha=0.85,
            zorder=5,
        )


def clean_continent(value):
    if pd.isna(value) or not str(value).strip():
        return "Other"
    value = str(value).strip()
    if value in CONTINENT_PIE_COLORS:
        return value
    return value


def top_continent_shares(route_table, positions, mode, direction, top_n=2):
    if route_table.empty:
        return []
    value_col = "route_value" if mode == "absolute" else "delta_route_value"
    table = route_table.copy()
    table = table[
        table["source_iso3"].isin(positions.index)
        & table["destination_iso3"].isin(positions.index)
    ].copy()
    if table.empty:
        return []
    table["pie_weight"] = table[value_col].abs()
    table = table[table["pie_weight"] > 0].copy()
    if table.empty:
        return []
    table["source_continent"] = table["source_iso3"].map(
        positions["continent"].map(clean_continent)
    )
    table["destination_continent"] = table["destination_iso3"].map(
        positions["continent"].map(clean_continent)
    )
    if direction == "inflow":
        node_col = "destination_iso3"
        continent_col = "source_continent"
    else:
        node_col = "source_iso3"
        continent_col = "destination_continent"

    node_totals = (
        table.groupby(node_col)["pie_weight"].sum().sort_values(ascending=False)
    )
    pies = []
    for node in node_totals.head(top_n).index:
        node_rows = table[table[node_col] == node]
        shares = node_rows.groupby(continent_col)["pie_weight"].sum()
        shares = shares[shares > 0].sort_values(ascending=False)
        total = float(shares.sum())
        if total <= 0:
            continue
        pies.append(
            {
                "node": node,
                "values": (shares / total).to_dict(),
            }
        )
    return pies


def add_pie_inset(ax, center_x, center_y, values, node, edge_color):
    labels = list(values.keys())
    sizes = [values[label] for label in labels]
    colors = [CONTINENT_PIE_COLORS.get(label, CONTINENT_PIE_COLORS["Other"]) for label in labels]
    inset = ax.inset_axes([center_x - 0.055, center_y - 0.055, 0.11, 0.11])
    wedges, _ = inset.pie(
        sizes,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"linewidth": 0},
    )
    for wedge in wedges:
        wedge.set_linewidth(0)
    inset.add_patch(
        plt.Circle((0, 0), 1.0, fill=False, edgecolor=edge_color, linewidth=1.0)
    )
    inset.set_aspect("equal")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.set_facecolor("none")
    ax.text(
        center_x,
        center_y - 0.072,
        node,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.5,
        color="#111827",
        zorder=20,
    )


def add_continent_pies(ax, route_table, positions, mode):
    inflow_pies = top_continent_shares(route_table, positions, mode, "inflow", top_n=2)
    outflow_pies = top_continent_shares(route_table, positions, mode, "outflow", top_n=2)
    if not inflow_pies and not outflow_pies:
        return
    ax.text(
        0.22,
        0.175,
        "Inflow source",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.5,
        color="#111827",
        zorder=20,
    )
    ax.text(
        0.78,
        0.175,
        "Outflow destination",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.5,
        color="#111827",
        zorder=20,
    )
    for pie, center_x in zip(inflow_pies, [0.16, 0.28]):
        add_pie_inset(ax, center_x, 0.105, pie["values"], pie["node"], "#2563EB")
    for pie, center_x in zip(outflow_pies, [0.72, 0.84]):
        add_pie_inset(ax, center_x, 0.105, pie["values"], pie["node"], "#DC2626")


def panel_stats(routes, policy):
    subset = routes[routes["policy_scenario"] == policy]
    total = subset["scrap_t"].sum()
    policy_cost = subset["policy_cost"].sum() if "policy_cost" in subset.columns else np.nan
    return total, policy_cost


def format_policy_cost(policy_cost):
    if pd.isna(policy_cost):
        return "Policy cost: n/a"
    return f"Policy cost: ${policy_cost / 1e6:.2f}M"


def route_access_loss(access_loss_table, policy):
    subset = access_loss_table[access_loss_table["policy_scenario"] == policy]
    return 0.0 if subset.empty else float(
        (
            -subset.loc[
                subset["access_delta_li_t"] < 0,
                "access_delta_li_t",
            ]
        ).sum()
    )


def format_route_access_loss(access_loss_li_t):
    return f"Route access loss: {access_loss_li_t / 1000.0:.2f} kt Li"


def figure_context_label(method, strategy):
    method_label = (
        "economic technology choice" if method.lower() == "all" else method
    )
    if strategy == "Strategy 3":
        return f"{method_label}, global source access"
    if strategy == "Strategy 2":
        return f"{method_label}, producer-country sources"
    return f"{method_label}, {strategy}"


def main():
    args = parse_args()
    plt.rcParams["font.family"] = "Arial"
    positions = load_positions()
    policies = parse_policies(args.policies)
    access_reference_policy = (
        args.route_access_reference_policy.strip() or args.baseline_policy
    )
    load_policies = list(dict.fromkeys(policies + [access_reference_policy]))
    routes = load_routes(
        args.year,
        args.method,
        args.strategy,
        Path(args.routes_file),
        load_policies,
        args.joint_routes,
    )
    if args.baseline_policy not in policies:
        raise ValueError("baseline-policy must be included in --policies.")
    if access_reference_policy not in set(routes["policy_scenario"]):
        raise ValueError("route-access-reference-policy is missing from the route table.")
    delta_table = build_delta_table(
        routes, policies, args.baseline_policy, args.flow_column, args.min_delta
    )
    access_loss_table = build_access_loss_table(
        routes,
        policies,
        access_reference_policy,
        args.route_access_loss_column,
        args.min_delta,
    )

    current_routes = aggregate_policy_routes(
        routes, args.baseline_policy, args.flow_column
    ).rename(
        columns={args.baseline_policy: "route_value"}
    )
    plot_data = {args.baseline_policy: current_routes}
    for policy in [item for item in policies if item != args.baseline_policy]:
        plot_data[policy] = delta_table[delta_table["policy_scenario"] == policy].copy()
    node_scale_max = 1.0
    for policy in policies:
        mode = "absolute" if policy == args.baseline_policy else "delta"
        node_values = top_route_node_values(
            plot_data[policy], positions, mode, args.top_routes
        )
        if node_values:
            node_scale_max = max(node_scale_max, max(node_values.values()))

    ncols = 3 if len(policies) > 4 else 2
    nrows = int(np.ceil(len(policies) / ncols))
    row_height = 5.1 if args.show_continent_pies else 4.5
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, row_height * nrows), dpi=300)
    axes = np.array(axes).flatten()
    current_total, _ = panel_stats(routes, args.baseline_policy)

    for ax, policy in zip(axes, policies):
        m = make_map(ax)
        mode = "absolute" if policy == args.baseline_policy else "delta"
        draw_routes(
            ax,
            m,
            plot_data[policy],
            positions,
            mode,
            args.top_routes,
            node_scale_max,
        )
        if args.show_continent_pies:
            add_continent_pies(ax, plot_data[policy], positions, mode)
        ax.set_title(POLICY_TITLES.get(policy, policy), fontsize=12, weight="bold", pad=8)
        policy_total, policy_cost = panel_stats(routes, policy)
        access_loss = route_access_loss(access_loss_table, policy)
        if policy == args.baseline_policy:
            caption = (
                f"{format_route_access_loss(access_loss)}\n"
                f"{format_policy_cost(policy_cost)}"
            )
        else:
            caption = (
                f"{format_route_access_loss(access_loss)}\n"
                f"{format_policy_cost(policy_cost)}"
            )
        ax.text(
            0.02,
            0.22 if args.show_continent_pies else 0.03,
            caption,
            transform=ax.transAxes,
            fontsize=9,
            color="#111827",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=3),
            zorder=10,
        )
    for ax in axes[len(policies) :]:
        ax.axis("off")

    route_handles = [
        mlines.Line2D([], [], color="#2563EB", linewidth=2.5, label="Current flow or reduced route"),
        mlines.Line2D([], [], color="#DC2626", linewidth=2.5, label="New or increased route"),
    ]
    node_size_handles = [
        mlines.Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="#111827",
            markersize=np.sqrt(12 + 120 * np.sqrt(min(value / node_scale_max, 1.0))),
            label=label,
        )
        for value, label in [
            (1000.0, "1 kt Li"),
            (5000.0, "5 kt Li"),
            (10000.0, "10 kt Li"),
        ]
    ]
    node_size_handles.insert(
        0,
        mlines.Line2D(
            [],
            [],
            linestyle="None",
            marker="",
            label="Country's total Li flow on shown routes:",
        ),
    )
    fig.legend(
        handles=route_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=2,
        frameon=False,
        fontsize=9.5,
    )
    fig.legend(
        handles=node_size_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.897),
        ncol=4,
        frameon=False,
        fontsize=8.5,
    )
    if args.show_continent_pies:
        pie_handles = [
            mpatches.Patch(color=color, label=f"Pie sector: {continent}")
            for continent, color in CONTINENT_PIE_COLORS.items()
        ]
        fig.legend(
            handles=pie_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.862),
            ncol=6,
            frameon=False,
            fontsize=8,
        )
    fig.suptitle(
        f"Policy-driven transfer-route changes ({args.year}, {figure_context_label(args.method, args.strategy)})",
        fontsize=15,
        weight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.82], h_pad=3.0)

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)

    delta_output = output.with_suffix(".route_deltas.csv")
    delta_table.to_csv(delta_output, index=False)
    print(f"Wrote {output}")
    print(f"Wrote {delta_output}")


def plot_single_policy_route_map(
    year,
    method,
    strategy,
    routes_file,
    policy,
    output,
    top_routes=70,
    joint_routes=True,
    flow_column="scrap_t",
):
    plt.rcParams["font.family"] = "Arial"
    positions = load_positions()
    routes = load_routes(
        year,
        method,
        strategy,
        Path(routes_file),
        [policy],
        joint_routes,
    )
    route_table = aggregate_policy_routes(routes, policy, flow_column).rename(
        columns={policy: "route_value"}
    )
    fig, ax = plt.subplots(1, 1, figsize=(10.8, 4.8), dpi=300)
    m = make_map(ax)
    draw_routes(ax, m, route_table, positions, "absolute", top_routes)
    total_flow = route_table["route_value"].sum()
    ax.set_title(
        f"{POLICY_TITLES.get(policy, policy)} routes",
        fontsize=13,
        weight="bold",
        pad=8,
    )
    ax.text(
        0.02,
        0.03,
        f"Cross-border flow: {total_flow / 1e6:.2f} Mt",
        transform=ax.transAxes,
        fontsize=9,
        color="#111827",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=3),
        zorder=10,
    )
    fig.suptitle(
        f"Route-access-unconstrained economic benchmark ({year}, {figure_context_label(method, strategy)})",
        fontsize=15,
        weight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    output = Path(output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
