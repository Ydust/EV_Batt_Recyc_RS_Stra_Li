from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_FILE = (
    ROOT
    / "Figure_data"
    / "joint_policy_technology"
    / "joint_policy_transport_technology_routes_reference_relaxed_2030_2040_2050_with_open.csv"
)
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


def route_loss_table(routes):
    rows = []
    route_rows = []
    real = routes[~routes["is_unprocessed"].astype(bool)].copy()
    for year in sorted(real["year"].unique()):
        year_data = real[real["year"] == year]
        reference = year_data[year_data["policy_scenario"] == REFERENCE_POLICY]
        reference_routes = reference.groupby(
            ["source_iso3", "destination_iso3"], as_index=True
        )[VALUE_COL].sum()

        for policy in POLICIES:
            policy_routes = year_data[year_data["policy_scenario"] == policy].groupby(
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
            comparison["route_access_loss_li_t"] = (
                -comparison["route_access_delta_li_t"].clip(upper=0.0)
            )
            loss = float(comparison["route_access_loss_li_t"].sum())
            rows.append(
                {
                    "year": int(year),
                    "policy_scenario": policy,
                    "route_access_loss_li_t": loss,
                    "route_access_loss_kt_li": loss / 1000.0,
                }
            )
            losses = comparison[comparison["route_access_loss_li_t"] > 0].copy()
            losses = losses.reset_index()
            losses["year"] = int(year)
            losses["policy_scenario"] = policy
            route_rows.append(losses)

    by_policy = pd.DataFrame(rows)
    by_route = pd.concat(route_rows, ignore_index=True)
    return by_policy, by_route


def plot(by_policy, by_route):
    plt.rcParams["font.family"] = "Arial"
    fig = plt.figure(figsize=(12.5, 6.6), dpi=300)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.28)
    ax = fig.add_subplot(grid[0, 0])
    ax_routes = fig.add_subplot(grid[0, 1])

    years = sorted(by_policy["year"].unique())
    x = np.arange(len(years))
    width = 0.18
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(POLICIES))
    for offset, policy in zip(offsets, POLICIES):
        subset = by_policy[by_policy["policy_scenario"] == policy].set_index("year")
        values = [subset.loc[year, "route_access_loss_kt_li"] for year in years]
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=POLICY_COLORS[policy],
            label=POLICY_LABELS[policy],
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, value in zip(bars, values):
            if value >= 100:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 18,
                    f"{value:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#111827",
                )

    ax.set_title("A. Route-access loss by policy scenario", loc="left", fontsize=12, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(year) for year in years])
    ax.set_ylabel("Route-access loss (kt Li)")
    ax.set_xlabel("Year")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper left", fontsize=9)

    focus_year = 2050
    focus_policy = "strict_policy"
    focus = by_route[
        (by_route["year"] == focus_year)
        & (by_route["policy_scenario"] == focus_policy)
    ].copy()
    focus["route"] = focus["source_iso3"] + " -> " + focus["destination_iso3"]
    top = focus.sort_values("route_access_loss_li_t", ascending=False).head(10)
    top = top.sort_values("route_access_loss_li_t")
    route_values = top["route_access_loss_li_t"] / 1000.0
    ax_routes.barh(
        top["route"],
        route_values,
        color="#F28E2B",
        edgecolor="white",
        linewidth=0.6,
    )
    for y_pos, value in enumerate(route_values):
        ax_routes.text(
            value + max(route_values) * 0.015,
            y_pos,
            f"{value:.1f}",
            va="center",
            fontsize=8,
            color="#111827",
        )
    ax_routes.set_title(
        "B. Largest lost benchmark routes\n(2050 strict policy)",
        loc="left",
        fontsize=12,
        weight="bold",
    )
    ax_routes.set_xlabel("Route-access loss (kt Li)")
    ax_routes.grid(axis="x", color="#E5E7EB", linewidth=0.8)
    ax_routes.set_axisbelow(True)
    ax_routes.spines["top"].set_visible(False)
    ax_routes.spines["right"].set_visible(False)

    fig.suptitle(
        "Route-access constraints shift accessible secondary lithium",
        fontsize=15,
        weight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "Route-access loss is calculated as the sum of route-level reductions in battery-embedded secondary Li relative to the route-access-unconstrained benchmark.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#374151",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    return fig


def main():
    routes = pd.read_csv(INPUT_FILE)
    by_policy, by_route = route_loss_table(routes)
    by_policy.to_csv(OUTPUT_DIR / "route_access_loss_by_policy_year.csv", index=False)
    by_route.to_csv(OUTPUT_DIR / "route_access_loss_by_route.csv", index=False)
    fig = plot(by_policy, by_route)
    png = OUTPUT_DIR / "route_access_loss_main_figure.png"
    pdf = OUTPUT_DIR / "route_access_loss_main_figure.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
