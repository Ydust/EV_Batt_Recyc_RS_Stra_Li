from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE_DIR = ROOT / "Figure_data" / "joint_policy_technology"
OUT_DIR = BASE_DIR / "pyrohydro_policy_robustness"
POLICY = "current_policy"
SCENARIOS = [
    ("S1", BASE_DIR / "pyrohydro_sensitivity_conservative_annual_gurobi"),
    ("S2", BASE_DIR / "pyrohydro_sensitivity_s2_annual_gurobi"),
    ("S3", BASE_DIR / "pyrohydro_sensitivity_medium_annual_gurobi"),
    ("S4", BASE_DIR / "pyrohydro_sensitivity_s4_annual_gurobi"),
    ("S5", BASE_DIR / "pyrohydro_sensitivity_s5_annual_gurobi"),
]
TECHNOLOGIES = ["Direct", "Hydro", "PyroHydro"]
REGION_ORDER = [
    "China",
    "United States",
    "European Union",
    "Japan/Korea",
    "Other developed",
    "Rest of world",
]
EU_ISO3 = {
    "AUT",
    "BEL",
    "BGR",
    "HRV",
    "CYP",
    "CZE",
    "DNK",
    "EST",
    "FIN",
    "FRA",
    "DEU",
    "GRC",
    "HUN",
    "IRL",
    "ITA",
    "LVA",
    "LTU",
    "LUX",
    "MLT",
    "NLD",
    "POL",
    "PRT",
    "ROU",
    "SVK",
    "SVN",
    "ESP",
    "SWE",
}
OTHER_DEVELOPED = {
    "AUS",
    "CAN",
    "CHE",
    "GBR",
    "ISL",
    "ISR",
    "NOR",
    "NZL",
    "SGP",
}


def region_group(iso3):
    if iso3 == "CHN":
        return "China"
    if iso3 == "USA":
        return "United States"
    if iso3 in EU_ISO3:
        return "European Union"
    if iso3 in {"JPN", "KOR"}:
        return "Japan/Korea"
    if iso3 in OTHER_DEVELOPED:
        return "Other developed"
    return "Rest of world"


def read_scenario(label, data_dir):
    path = data_dir / "annual_dynamic_country_technology_radar.csv"
    if path.exists():
        data = pd.read_csv(path)
        data = data[data["policy_scenario"] == POLICY].copy()
        data = data[data["technology"].isin(TECHNOLOGIES)].copy()
        data["year"] = pd.to_numeric(data["year"], errors="coerce").astype(int)
        data["recovered_lithium_t"] = pd.to_numeric(data["recovered_lithium_t"], errors="coerce").fillna(0.0)
        data["region_group"] = data["region"].map(region_group)
        grouped = data.groupby(["year", "region_group", "technology"], as_index=False).agg(
            recovered_lithium_t=("recovered_lithium_t", "sum")
        )
    else:
        routes = pd.read_csv(data_dir / "dynamic_scale_routes.csv")
        routes = routes[
            (routes["policy_scenario"] == POLICY)
            & (routes["technology"].isin(TECHNOLOGIES))
            & (~routes["is_unprocessed"].astype(bool))
        ].copy()
        routes["year"] = pd.to_numeric(routes["year"], errors="coerce").astype(int)
        routes["recovered_lithium_t"] = pd.to_numeric(
            routes["recovered_lithium_t"], errors="coerce"
        ).fillna(0.0)
        routes["region_group"] = routes["destination_iso3"].map(region_group)
        grouped = routes.groupby(["year", "region_group", "technology"], as_index=False).agg(
            recovered_lithium_t=("recovered_lithium_t", "sum")
        )
    totals = grouped.groupby(["year", "region_group"], as_index=False).agg(
        total_recovered_lithium_t=("recovered_lithium_t", "sum")
    )
    grouped = grouped.merge(totals, on=["year", "region_group"], how="left")
    grouped["technology_share_pct"] = (
        grouped["recovered_lithium_t"] / grouped["total_recovered_lithium_t"] * 100.0
    ).fillna(0.0)
    grouped["scenario"] = label
    return grouped


def build_share_band():
    frames = [read_scenario(label, path) for label, path in SCENARIOS]
    data = pd.concat(frames, ignore_index=True)
    years = sorted(data["year"].unique())
    index = pd.MultiIndex.from_product(
        [[label for label, _ in SCENARIOS], years, REGION_ORDER, TECHNOLOGIES],
        names=["scenario", "year", "region_group", "technology"],
    )
    data = (
        data.set_index(["scenario", "year", "region_group", "technology"])
        .reindex(index, fill_value=0.0)
        .reset_index()
    )
    band = data.groupby(["year", "region_group", "technology"], as_index=False).agg(
        mean_share_pct=("technology_share_pct", "mean"),
        min_share_pct=("technology_share_pct", "min"),
        max_share_pct=("technology_share_pct", "max"),
    )
    return band


def plot_current_policy_region_heatmap(band):
    years = sorted(band["year"].unique())
    plt.rcParams.update({"font.family": "Arial"})
    fig, axes = plt.subplots(
        len(REGION_ORDER),
        1,
        figsize=(10.8, 8.2),
        sharex=True,
        dpi=300,
        gridspec_kw={"hspace": 0.10},
    )
    image = None
    for ax, region in zip(axes, REGION_ORDER):
        table = (
            band[band["region_group"] == region]
            .pivot_table(
                index="technology",
                columns="year",
                values="mean_share_pct",
                aggfunc="first",
                fill_value=0.0,
            )
            .reindex(index=TECHNOLOGIES, columns=years, fill_value=0.0)
        )
        image = ax.imshow(
            table.values,
            aspect="auto",
            cmap="YlGnBu",
            vmin=0,
            vmax=100,
            extent=[min(years) - 0.5, max(years) + 0.5, len(TECHNOLOGIES) - 0.5, -0.5],
        )
        ax.set_yticks(range(len(TECHNOLOGIES)))
        ax.set_yticklabels(TECHNOLOGIES, fontsize=8.5)
        ax.set_ylabel(region, rotation=0, ha="right", va="center", labelpad=54, fontsize=10)
        ax.tick_params(axis="both", length=0, labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
            spine.set_color("#374151")
    axes[-1].set_xticks([2025, 2030, 2035, 2040, 2045, 2050])
    axes[-1].set_xlabel("Year", fontsize=10)
    cbar = fig.colorbar(image, ax=axes, orientation="horizontal", fraction=0.035, pad=0.06, aspect=45)
    cbar.set_label("Mean technology share across S1-S5 (%)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    fig.suptitle(
        "Regional technology-mix sketch under Current policy",
        fontsize=13,
        y=0.985,
    )
    fig.text(
        0.5,
        0.025,
        "Rows are destination-region groups. Each heat strip shows Direct, Hydro, and PyroHydro shares over time; S1-S5 are averaged for this sketch.",
        ha="center",
        fontsize=8.6,
        color="#374151",
    )
    fig.subplots_adjust(left=0.19, right=0.985, bottom=0.14, top=0.92)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "regional_technology_share_current_policy_sketch.png"
    pdf = OUT_DIR / "regional_technology_share_current_policy_sketch.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


def main():
    band = build_share_band()
    band.to_csv(OUT_DIR / "regional_technology_share_current_policy_sketch.csv", index=False)
    plot_current_policy_region_heatmap(band)


if __name__ == "__main__":
    main()
