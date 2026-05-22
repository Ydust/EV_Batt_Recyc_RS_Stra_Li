import argparse
from pathlib import Path

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "Figure_data" / "joint_policy_technology" / "annual_dynamic_avail06_direct_x1p2"
DEFAULT_OUT_DIR = ROOT / "Figure_data" / "joint_policy_technology" / "annual_dynamic_avail06_direct_x1p2"
POLICY_ORDER = ["reference_policy", "current_policy", "strict_policy", "critical_route_policy"]
POLICY_LABELS = {
    "reference_policy": "Reference",
    "current_policy": "Current",
    "strict_policy": "Strict",
    "critical_route_policy": "Critical-route",
}
POLICY_COLORS = {
    "reference_policy": "#0072B2",
    "current_policy": "#009E73",
    "strict_policy": "#CC79A7",
    "critical_route_policy": "#D55E00",
}
TECHNOLOGY_ORDER = ["Direct", "Hydro", "Pyro", "PyroHydro"]
TECHNOLOGY_LINESTYLES = {
    "Direct": "-",
    "Hydro": "--",
    "Pyro": ":",
    "PyroHydro": "-.",
}
TECHNOLOGY_MARKERS = {
    "Direct": "o",
    "Hydro": "s",
    "Pyro": "^",
    "PyroHydro": "D",
}
TECHNOLOGY_POINT_OFFSET = {
    "Direct": 1.5,
    "Hydro": 0.5,
    "Pyro": -0.5,
    "PyroHydro": -1.5,
}
REGION_POINT_OFFSET = {
    "Global": 0.0,
    "CHN": 1.8,
    "KOR": 0.9,
    "IND": -0.9,
    "USA": -1.8,
}
RADAR_YEARS = list(range(2030, 2051))
RADAR_COUNTRIES = {
    "Global": "Global",
    "CHN": "China",
    "KOR": "Korea",
    "IND": "India",
    "USA": "USA",
}
RADAR_COLORS = {
    "Global": "#222222",
    "CHN": "#E69F00",
    "KOR": "#56B4E9",
    "IND": "#009E73",
    "USA": "#CC79A7",
}
TREND_WINDOW_YEARS = 3


def smooth_trend(frame, value_col):
    frame = frame.sort_values("year").copy()
    frame[f"{value_col}_raw"] = frame[value_col].astype(float)
    frame[value_col] = (
        frame[value_col]
        .astype(float)
        .rolling(TREND_WINDOW_YEARS, center=True, min_periods=1)
        .median()
    )
    return frame


def build_plot_data(data):
    corrected_frames = []
    for (policy, technology), group in data.groupby(["policy_scenario", "technology"]):
        corrected = smooth_trend(group, "recovered_lithium_t")
        corrected_frames.append(corrected)
    corrected = pd.concat(corrected_frames, ignore_index=True)
    totals = (
        corrected.groupby(["year", "policy_scenario"], as_index=False)["recovered_lithium_t"]
        .sum()
        .rename(columns={"recovered_lithium_t": "total_recovered_lithium_t"})
    )
    corrected = corrected.merge(totals, on=["year", "policy_scenario"], how="left")
    corrected["technology_share_pct_raw"] = corrected["technology_share_pct"]
    corrected["technology_share_pct"] = (
        corrected["recovered_lithium_t"]
        / corrected["total_recovered_lithium_t"]
        * 100.0
    ).fillna(0.0)

    return corrected


def build_country_radar_data(routes):
    is_unprocessed = routes["is_unprocessed"]
    if is_unprocessed.dtype == object:
        is_unprocessed = is_unprocessed.astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        is_unprocessed = is_unprocessed.astype(bool)
    real = routes[~is_unprocessed].copy()
    real = real[real["technology"].isin(TECHNOLOGY_ORDER)].copy()
    country_rows = (
        real.groupby(
            ["year", "policy_scenario", "destination_iso3", "technology"],
            as_index=False,
        )["recovered_lithium_t"]
        .sum()
        .rename(columns={"destination_iso3": "region"})
    )
    global_rows = (
        real.groupby(["year", "policy_scenario", "technology"], as_index=False)[
            "recovered_lithium_t"
        ]
        .sum()
        .assign(region="Global")
    )
    radar = pd.concat([country_rows, global_rows], ignore_index=True)
    radar = radar[radar["region"].isin(RADAR_COUNTRIES)].copy()

    smoothed = []
    for _, group in radar.groupby(["policy_scenario", "region", "technology"]):
        smoothed.append(smooth_trend(group, "recovered_lithium_t"))
    radar = pd.concat(smoothed, ignore_index=True)
    totals = (
        radar.groupby(["year", "policy_scenario", "region"], as_index=False)[
            "recovered_lithium_t"
        ]
        .sum()
        .rename(columns={"recovered_lithium_t": "total_recovered_lithium_t"})
    )
    radar = radar.merge(totals, on=["year", "policy_scenario", "region"], how="left")
    radar["technology_share_pct"] = (
        radar["recovered_lithium_t"] / radar["total_recovered_lithium_t"] * 100.0
    ).fillna(0.0)
    return radar


def plot_country_radar(radar, out_dir):
    plt.rcParams.update({"font.family": "Arial"})
    technology_order = [
        technology
        for technology in TECHNOLOGY_ORDER
        if technology in set(radar["technology"].dropna())
    ]
    angles = np.linspace(0, 2 * np.pi, len(RADAR_YEARS), endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 14),
        subplot_kw={"projection": "polar"},
        dpi=300,
    )
    circle_angles = np.linspace(0, 2 * np.pi, 240)
    radial_label_angle = np.deg2rad(160)

    for ax, policy in zip(axes.flat, POLICY_ORDER):
        subset = radar[radar["policy_scenario"] == policy].copy()
        for region in RADAR_COUNTRIES:
            for technology in technology_order:
                line = (
                    subset[
                        (subset["region"] == region)
                        & (subset["technology"] == technology)
                    ]
                    .set_index("year")
                    .reindex(RADAR_YEARS)
                )
                values = line["technology_share_pct"].fillna(0.0).to_numpy()
                closed_values = np.concatenate([values, values[:1]])
                ax.plot(
                    closed_angles,
                    closed_values,
                    color="0.25",
                    linestyle=TECHNOLOGY_LINESTYLES[technology],
                    linewidth=2.2 if region == "Global" else 1.0,
                    alpha=0.9 if region == "Global" else 0.28,
                )
                ax.scatter(
                    angles,
                    np.clip(
                        values
                        + REGION_POINT_OFFSET[region]
                        + TECHNOLOGY_POINT_OFFSET[technology],
                        0,
                        100,
                    ),
                    s=42 if region == "Global" else 26,
                    marker=TECHNOLOGY_MARKERS[technology],
                    facecolors="none",
                    edgecolors=RADAR_COLORS[region],
                    linewidths=1.6 if region == "Global" else 1.2,
                    alpha=0.95 if region == "Global" else 0.78,
                )
        ax.text(
            np.deg2rad(90),
            72,
            POLICY_LABELS[policy],
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=18,
            fontweight="bold",
            color="black",
        )
        ax.set_xticks(angles)
        ax.set_xticklabels([str(year) for year in RADAR_YEARS], fontsize=8)
        ax.set_ylim(-5, 100)
        ax.set_yticks([])
        ax.grid(False)
        ax.spines["polar"].set_visible(False)
        for radius, label, line_width in [
            (0, "0%", 1.0),
            (25, "25%", 1.0),
            (50, "50%", 1.0),
            (75, "75%", 1.0),
            (100, "100%", 1.8),
        ]:
            ax.plot(
                circle_angles,
                np.ones_like(circle_angles) * radius,
                linestyle=(0, (5, 3)),
                color="black",
                linewidth=line_width,
                alpha=0.85,
            )
            ax.text(
                radial_label_angle,
                radius,
                label,
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=10,
                fontweight="bold",
                color="black",
                rotation=70,
            )

    country_handles = [
        Line2D(
            [0],
            [0],
            color=RADAR_COLORS[region],
            marker="o",
            markersize=7,
            markerfacecolor="none",
            markeredgewidth=1.5,
            linestyle="None",
            label=label,
        )
        for region, label in RADAR_COUNTRIES.items()
    ]
    technology_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            lw=2.2,
            linestyle=TECHNOLOGY_LINESTYLES[technology],
            marker=TECHNOLOGY_MARKERS[technology],
            markerfacecolor="none",
            markersize=6,
            label=technology,
        )
        for technology in technology_order
    ]
    fig.legend(
        handles=country_handles,
        loc="upper center",
        ncol=len(country_handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.965),
        title="Region (point color)",
        prop={"size": 13},
        title_fontsize=14,
    )
    fig.legend(
        handles=technology_handles,
        loc="lower center",
        ncol=len(technology_handles),
        frameon=False,
        bbox_to_anchor=(0.5, 0.035),
        title="Technology (line style)",
        prop={"size": 13},
        title_fontsize=14,
    )
    fig.suptitle(
        "Technology substitution profiles by policy scenario",
        y=1.01,
        fontsize=18,
    )
    fig.subplots_adjust(wspace=0.35, hspace=0.35, top=0.9, bottom=0.11)
    png = out_dir / "annual_dynamic_country_technology_radar.png"
    pdf = out_dir / "annual_dynamic_country_technology_radar.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir) if args.out_dir else data_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_dir / "dynamic_scale_summary.csv")
    for column in ["year", "technology_share_pct", "route_modeled_cost", "recovered_lithium_t"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = build_plot_data(data)
    technology_order = [
        technology
        for technology in TECHNOLOGY_ORDER
        if technology in set(data["technology"].dropna())
    ]
    data.to_csv(out_dir / "annual_dynamic_trend_corrected.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for policy in POLICY_ORDER:
        for technology in technology_order:
            subset = data[
                (data["policy_scenario"] == policy)
                & (data["technology"] == technology)
            ].sort_values("year")
            axes[0].plot(
                subset["year"],
                subset["technology_share_pct"],
                color=POLICY_COLORS[policy],
                linewidth=2 if technology == "Direct" else 1.5,
                linestyle=TECHNOLOGY_LINESTYLES[technology],
                alpha=1.0 if technology == "Direct" else 0.75,
            )
    axes[0].set_title("Technology share by policy scenario")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Technology share (%)")
    axes[0].set_xlim(2025, 2050)
    axes[0].set_xticks([2025, 2030, 2035, 2040, 2045, 2050])
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis="y", color="0.9")

    for policy in POLICY_ORDER:
        for technology in technology_order:
            subset = data[
                (data["policy_scenario"] == policy)
                & (data["technology"] == technology)
            ].sort_values("year")
            axes[1].plot(
                subset["year"],
                subset["recovered_lithium_t"] / 1000,
                color=POLICY_COLORS[policy],
                linewidth=2 if technology == "Direct" else 1.3,
                linestyle=TECHNOLOGY_LINESTYLES[technology],
                alpha=1.0 if technology == "Direct" else 0.75,
            )
    axes[1].set_title("Recovered Li by technology")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Recovered Li (kt)")
    axes[1].set_xlim(2025, 2050)
    axes[1].set_xticks([2025, 2030, 2035, 2040, 2045, 2050])
    axes[1].grid(axis="y", color="0.9")

    policy_handles = [
        Line2D([0], [0], color=POLICY_COLORS[p], lw=2.2, label=POLICY_LABELS[p])
        for p in POLICY_ORDER
    ]
    technology_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            lw=2.2,
            linestyle=TECHNOLOGY_LINESTYLES[technology],
            label=technology,
        )
        for technology in TECHNOLOGY_ORDER
        if technology in technology_order
    ]
    fig.legend(
        handles=policy_handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.94),
        title="Policy scenario (line color)",
    )
    axes[1].legend(
        handles=technology_handles,
        loc="upper left",
        frameon=False,
        title="Technology (line style)",
    )
    title = "Annual technology mix under dynamic scale cost, capability mask, and Direct cost penalty"
    if "PyroHydro" in technology_order:
        title = f"{title}, with PyroHydro"
    fig.suptitle(title, y=1.03, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    png = out_dir / "annual_dynamic_direct_share_trend.png"
    pdf = out_dir / "annual_dynamic_direct_share_trend.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")

    routes = pd.read_csv(
        data_dir / "dynamic_scale_routes.csv",
        usecols=[
            "year",
            "policy_scenario",
            "destination_iso3",
            "technology",
            "recovered_lithium_t",
            "is_unprocessed",
        ],
    )
    for column in ["year", "recovered_lithium_t"]:
        routes[column] = pd.to_numeric(routes[column], errors="coerce")
    radar = build_country_radar_data(routes)
    radar.to_csv(out_dir / "annual_dynamic_country_technology_radar.csv", index=False)
    plot_country_radar(radar, out_dir)


if __name__ == "__main__":
    main()
