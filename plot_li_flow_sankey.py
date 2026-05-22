from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parent
ROUTES_FILE = (
    ROOT / "Figure_data" / "joint_policy_technology" / "lithium_loss_scenarios"
    / "lithium_loss_scenarios_routes.csv"
)
COUNTRIES_FILE = ROOT / "all_countries.csv"
OUT_DIR = ROOT / "Figure_data" / "joint_policy_technology" / "lithium_loss_scenarios"

YEAR = 2050
MITIGATION = "baseline"
POLICIES = ["current_policy", "reference_policy", "strict_policy", "critical_route_policy"]
POLICY_LABELS = {
    "current_policy":        "Current policy",
    "reference_policy":      "Reference policy",
    "strict_policy":         "Strict policy",
    "critical_route_policy": "Critical-route policy",
}
STRATEGY = "Strategy 3"

CONTINENT_COLORS = {
    "Asia":    "#0072B2",
    "Europe":  "#009E73",
    "America": "#E69F00",
    "Africa":  "#CC79A7",
    "Pacific": "#56B4E9",
    "Oceania": "#56B4E9",
    "Other":   "#999999",
}


def load_filtered_routes(policy):
    routes = pd.read_csv(ROUTES_FILE)
    routes = routes[
        (routes["year"] == YEAR)
        & (routes["mitigation_scenario"] == MITIGATION)
        & (routes["policy_scenario"] == policy)
        & (routes["strategy"] == STRATEGY)
        & (~routes["is_unprocessed"].astype(bool))
    ].copy()
    countries = pd.read_csv(COUNTRIES_FILE).set_index("iso3")
    routes["src_continent"] = routes["source_iso3"].map(countries["continent"]).fillna("Other")
    routes["dst_continent"] = routes["destination_iso3"].map(countries["continent"]).fillna("Other")
    routes["src_country"] = routes["source_iso3"].map(countries["country"]).fillna(routes["source_iso3"])
    routes["dst_country"] = routes["destination_iso3"].map(countries["country"]).fillna(routes["destination_iso3"])
    return routes


def aggregate_by_continent(routes):
    agg = routes.groupby(["src_continent", "dst_continent"], as_index=False)[
        "recovered_lithium_t"
    ].sum()
    agg = agg[agg["recovered_lithium_t"] > 0].copy()
    agg["recovered_lithium_kt"] = agg["recovered_lithium_t"] / 1000.0
    return agg


def aggregate_by_country(routes, top_n_src=8, top_n_dst=8):
    src_top = (
        routes.groupby("src_country")["recovered_lithium_t"].sum()
        .sort_values(ascending=False).head(top_n_src).index.tolist()
    )
    dst_top = (
        routes.groupby("dst_country")["recovered_lithium_t"].sum()
        .sort_values(ascending=False).head(top_n_dst).index.tolist()
    )
    sub = routes.copy()
    sub["src_label"] = sub["src_country"].where(sub["src_country"].isin(src_top), "Other src")
    sub["dst_label"] = sub["dst_country"].where(sub["dst_country"].isin(dst_top), "Other dst")
    agg = sub.groupby(["src_label", "dst_label"], as_index=False)["recovered_lithium_t"].sum()
    agg = agg[agg["recovered_lithium_t"] > 0].copy()
    agg["recovered_lithium_kt"] = agg["recovered_lithium_t"] / 1000.0
    src_order = src_top + (["Other src"] if "Other src" in agg["src_label"].values else [])
    dst_order = dst_top + (["Other dst"] if "Other dst" in agg["dst_label"].values else [])

    countries_meta = pd.read_csv(COUNTRIES_FILE)
    name_to_continent = dict(zip(countries_meta["country"], countries_meta["continent"]))
    src_continent = {n: name_to_continent.get(n, "Other") for n in src_order}
    dst_continent = {n: name_to_continent.get(n, "Other") for n in dst_order}
    return agg, src_order, dst_order, src_continent, dst_continent


def draw_sankey(
    ax, links, src_order, dst_order, src_color_map, dst_color_map,
    title="", x_left=0.05, x_right=0.95, gap=0.012,
):
    src_totals = links.groupby(links.columns[0])[links.columns[2]].sum()
    dst_totals = links.groupby(links.columns[1])[links.columns[2]].sum()
    grand_total = float(src_totals.sum())

    n_src = len(src_order)
    n_dst = len(dst_order)
    src_gap_total = (n_src - 1) * gap
    dst_gap_total = (n_dst - 1) * gap
    src_avail = 1.0 - src_gap_total
    dst_avail = 1.0 - dst_gap_total

    src_y = {}
    cur = 1.0
    for s in src_order:
        h = src_totals.get(s, 0.0) / grand_total * src_avail
        src_y[s] = (cur - h, cur)
        cur -= h + gap

    dst_y = {}
    cur = 1.0
    for d in dst_order:
        h = dst_totals.get(d, 0.0) / grand_total * dst_avail
        dst_y[d] = (cur - h, cur)
        cur -= h + gap

    src_running = {s: src_y[s][1] for s in src_order}
    dst_running = {d: dst_y[d][1] for d in dst_order}

    src_col, tgt_col, val_col = links.columns[0], links.columns[1], links.columns[2]
    sorted_links = links.sort_values(val_col, ascending=False)

    for _, row in sorted_links.iterrows():
        s, d, v = row[src_col], row[tgt_col], row[val_col]
        if v <= 0:
            continue
        h = v / grand_total * src_avail
        h_dst = v / grand_total * dst_avail
        s_top = src_running[s]
        s_bot = s_top - h
        src_running[s] = s_bot
        d_top = dst_running[d]
        d_bot = d_top - h_dst
        dst_running[d] = d_bot

        x_mid = (x_left + x_right) / 2.0
        path_data = [
            (MplPath.MOVETO, (x_left, s_top)),
            (MplPath.CURVE4, (x_mid, s_top)),
            (MplPath.CURVE4, (x_mid, d_top)),
            (MplPath.CURVE4, (x_right, d_top)),
            (MplPath.LINETO, (x_right, d_bot)),
            (MplPath.CURVE4, (x_mid, d_bot)),
            (MplPath.CURVE4, (x_mid, s_bot)),
            (MplPath.CURVE4, (x_left, s_bot)),
            (MplPath.CLOSEPOLY, (x_left, s_top)),
        ]
        codes, verts = zip(*path_data)
        path = MplPath(verts, codes)
        color = src_color_map.get(s, "#888888")
        patch = PathPatch(path, facecolor=color, edgecolor="none", alpha=0.45, linewidth=0)
        ax.add_patch(patch)

    node_w = 0.012
    for s in src_order:
        y0, y1 = src_y[s]
        rect = Rectangle((x_left - node_w, y0), node_w, y1 - y0,
                         facecolor=src_color_map.get(s, "#888888"),
                         edgecolor="white", linewidth=0.6)
        ax.add_patch(rect)
        kt = src_totals.get(s, 0.0) / 1000.0
        ax.text(x_left - node_w - 0.01, (y0 + y1) / 2.0,
                f"{s}  {kt:,.0f} kt", ha="right", va="center", fontsize=9)

    for d in dst_order:
        y0, y1 = dst_y[d]
        rect = Rectangle((x_right, y0), node_w, y1 - y0,
                         facecolor=dst_color_map.get(d, "#888888"),
                         edgecolor="white", linewidth=0.6)
        ax.add_patch(rect)
        kt = dst_totals.get(d, 0.0) / 1000.0
        ax.text(x_right + node_w + 0.01, (y0 + y1) / 2.0,
                f"{d}  {kt:,.0f} kt", ha="left", va="center", fontsize=9)

    ax.set_xlim(-0.42, 1.42)
    ax.set_ylim(-0.02, 1.05)
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11.5, pad=8, loc="left")


def main():
    plt.rcParams.update({"font.family": "Arial"})

    fig, axes = plt.subplots(2, 2, figsize=(15.6, 12.5), dpi=300)
    panel_letters = [["a", "b"], ["c", "d"]]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for idx, policy in enumerate(POLICIES):
        r, c = idx // 2, idx % 2
        routes = load_filtered_routes(policy)
        cou_links, cou_src_order, cou_dst_order, src_cc, dst_cc = (
            aggregate_by_country(routes, top_n_src=8, top_n_dst=8)
        )
        src_country_color = {n: CONTINENT_COLORS.get(c, "#999999") for n, c in src_cc.items()}
        dst_country_color = {n: CONTINENT_COLORS.get(c, "#999999") for n, c in dst_cc.items()}
        src_country_color["Other src"] = "#BBBBBB"
        dst_country_color["Other dst"] = "#BBBBBB"

        draw_sankey(
            axes[r, c],
            cou_links[["src_label", "dst_label", "recovered_lithium_t"]],
            cou_src_order, cou_dst_order,
            src_color_map=src_country_color, dst_color_map=dst_country_color,
            title=f"{panel_letters[r][c]}  {POLICY_LABELS[policy]}",
        )

        cou_links.to_csv(OUT_DIR / f"li_flow_country_top8_{YEAR}_{policy}.csv", index=False)

    legend_continents = ["Asia", "Europe", "America", "Africa", "Pacific"]
    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor=CONTINENT_COLORS[c], edgecolor="white", label=c)
        for c in legend_continents
    ] + [
        Rectangle((0, 0), 1, 1, facecolor="#BBBBBB", edgecolor="white", label="Other (aggregated)")
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=len(legend_handles),
        frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.025), title="Continent (color of source / destination)",
        title_fontsize=10,
    )

    fig.suptitle(
        f"Recovered lithium country-level flows under four policies ({YEAR})",
        y=0.985, fontsize=13.5, fontweight="bold",
    )
    fig.text(
        0.5, 0.005,
        "Top-8 source countries on left, top-8 destination countries on right (others aggregated). "
        "Ribbon thickness ~ recovered Li (t). Color = source-country continent.",
        ha="center", fontsize=9, color="#374151",
    )
    fig.subplots_adjust(left=0.04, right=0.985, top=0.93, bottom=0.10, hspace=0.12, wspace=0.10)

    png = OUT_DIR / f"li_flow_sankey_country_{YEAR}_4policies.png"
    pdf = OUT_DIR / f"li_flow_sankey_country_{YEAR}_4policies.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
