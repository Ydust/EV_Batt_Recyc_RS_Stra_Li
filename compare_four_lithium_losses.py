from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
CAPTURE_LOSS_FILE = ROOT / "Figure_data" / "capture_loss_with_manufacturing_global.csv"
LITHIUM_LOSS_SCENARIO_FILE = (
    ROOT
    / "Figure_data"
    / "joint_policy_technology"
    / "lithium_loss_scenarios"
    / "lithium_loss_scenarios_summary.csv"
)
TECH_CHOICE_FILE = (
    ROOT
    / "trans"
    / "scenario_result"
    / "high_collection"
    / "baseline"
    / "technology_choice_modes"
    / "technology_choice_mode_summary.csv"
)
OUTPUT_DIR = ROOT / "Figure_data"
OUTPUT_DATA = OUTPUT_DIR / "lithium_loss_comparison.csv"
OUTPUT_FIGURE = OUTPUT_DIR / "lithium_loss_comparison.svg"
LEGACY_OUTPUT_DATA = OUTPUT_DIR / "four_lithium_loss_comparison.csv"
LEGACY_OUTPUT_FIGURE = OUTPUT_DIR / "four_lithium_loss_comparison.svg"

YEARS = [2025, 2030, 2035, 2040, 2045, 2050]
COLLECTION_SCENARIO = "high_collection"
STRATEGY = "Strategy 3"
TRADE_POLICY_SCENARIO = "current_policy"
REALISTIC_CHOICE_MODE = "Realistic_multiobjective"
OPTIMAL_CHOICE_MODE = "Optimal_lithium"


def load_capture_losses():
    capture = pd.read_csv(CAPTURE_LOSS_FILE)
    capture = capture[
        (capture["scenario"] == COLLECTION_SCENARIO) & (capture["Year"].isin(YEARS))
    ].copy()
    return capture[
        [
            "Year",
            "eol_capture_loss_lithium_t",
            "manufacturing_capture_loss_lithium_t",
        ]
    ].rename(columns={"Year": "year"})


def load_technology_and_policy_loss():
    losses = pd.read_csv(LITHIUM_LOSS_SCENARIO_FILE)
    losses = losses[
        (losses["mitigation_scenario"] == "baseline")
        & (losses["policy_scenario"] == TRADE_POLICY_SCENARIO)
        & (losses["strategy"] == STRATEGY)
        & (losses["year"].isin(YEARS))
    ].copy()
    return losses[
        [
            "year",
            "technology_recovery_loss_t",
            "unprocessed_lithium_t",
            "route_access_loss_t",
            "route_access_displaced_lithium_t",
        ]
    ].rename(
        columns={
            "route_access_loss_t": "final_route_access_loss_t",
            "route_access_displaced_lithium_t": "trade_policy_loss_t",
        }
    )


def load_economic_selection_loss():
    choices = pd.read_csv(TECH_CHOICE_FILE)
    realistic = choices[
        (choices["Strategy type"] == STRATEGY)
        & (choices["choice_mode"] == REALISTIC_CHOICE_MODE)
        & (choices["year"].isin(YEARS))
    ][["year", "recycled_lithium"]].rename(
        columns={"recycled_lithium": "realistic_recycled_lithium_t"}
    )
    optimal = choices[
        (choices["Strategy type"] == STRATEGY)
        & (choices["choice_mode"] == OPTIMAL_CHOICE_MODE)
        & (choices["year"].isin(YEARS))
    ][["year", "recycled_lithium"]].rename(
        columns={"recycled_lithium": "optimal_recycled_lithium_t"}
    )
    data = realistic.merge(optimal, on="year", how="outer")
    data["economic_selection_loss_t"] = (
        data["optimal_recycled_lithium_t"] - data["realistic_recycled_lithium_t"]
    ).clip(lower=0)
    return data[["year", "economic_selection_loss_t"]]


def build_comparison():
    data = load_capture_losses()
    data = data.merge(load_technology_and_policy_loss(), on="year", how="outer")
    data = data.merge(load_economic_selection_loss(), on="year", how="outer")
    data = data.fillna(0.0).sort_values("year")
    data["capture_loss_lithium_t"] = (
        data["eol_capture_loss_lithium_t"]
        + data["manufacturing_capture_loss_lithium_t"]
    )
    data["total_compared_loss_t"] = (
        data["capture_loss_lithium_t"]
        + data["technology_recovery_loss_t"]
        + data["trade_policy_loss_t"]
        + data["economic_selection_loss_t"]
    )
    kt_cols = [column for column in data.columns if column.endswith("_t")]
    for column in kt_cols:
        data[column.replace("_t", "_kt")] = data[column] / 1000.0
    return data


def write_svg(data):
    width = 1060
    height = 620
    left = 78
    right = 34
    top = 72
    bottom = 86
    plot_w = width - left - right
    plot_h = height - top - bottom
    series = [
        ("capture_loss_lithium_kt", "Capture loss", "#D1495B"),
        ("technology_recovery_loss_kt", "Technology recovery loss", "#F28E2B"),
        ("trade_policy_loss_kt", "Trade-policy loss", "#2A9D8F"),
        ("economic_selection_loss_kt", "Economic / choice loss", "#6D5BD0"),
    ]
    y_max = max(float(data[[item[0] for item in series]].max().max()) * 1.18, 1.0)
    groups = len(data)
    group_w = plot_w / groups
    bar_w = group_w * 0.16
    offsets = [-1.8 * bar_w, -0.6 * bar_w, 0.6 * bar_w, 1.8 * bar_w]

    def y_scale(value):
        return top + plot_h - float(value) / y_max * plot_h

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<text x="78" y="38" font-family="Arial, sans-serif" font-size="21" font-weight="700" fill="#263238">Lithium Loss Categories Compared</text>',
        '<text x="78" y="58" font-family="Arial, sans-serif" font-size="12" fill="#5F6B73">Capture loss combines EOL and manufacturing scrap capture losses. Trade-policy loss uses current-policy displaced-route lithium. Units: kt Li.</text>',
    ]

    for i in range(6):
        tick = y_max * i / 5
        y = y_scale(tick)
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#D6D8DC" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#5F6B73" text-anchor="end">{tick:.0f}</text>'
        )

    for idx, row in enumerate(data.itertuples(index=False)):
        center = left + group_w * idx + group_w / 2
        elements.append(
            f'<text x="{center:.1f}" y="{top + plot_h + 28}" font-family="Arial, sans-serif" font-size="12" fill="#455A64" text-anchor="middle">{int(row.year)}</text>'
        )
        for offset, (column, label, color) in zip(offsets, series):
            value = float(getattr(row, column))
            x = center + offset - bar_w / 2
            y = y_scale(value)
            h = top + plot_h - y
            elements.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>'
            )

    elements.extend(
        [
            f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#455A64" stroke-width="1.2"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#455A64" stroke-width="1.2"/>',
        ]
    )

    legend_x = 680
    legend_y = 82
    for i, (_, label, color) in enumerate(series):
        y = legend_y + i * 22
        elements.append(f'<rect x="{legend_x}" y="{y - 10}" width="16" height="12" fill="{color}"/>')
        elements.append(
            f'<text x="{legend_x + 24}" y="{y}" font-family="Arial, sans-serif" font-size="13" fill="#263238">{label}</text>'
        )
    elements.append("</svg>")
    OUTPUT_FIGURE.write_text("\n".join(elements), encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = build_comparison()
    data.to_csv(OUTPUT_DATA, index=False)
    write_svg(data)
    data.to_csv(LEGACY_OUTPUT_DATA, index=False)
    LEGACY_OUTPUT_FIGURE.write_text(OUTPUT_FIGURE.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Wrote {OUTPUT_DATA}")
    print(f"Wrote {OUTPUT_FIGURE}")
    print(
        data[
            [
                "year",
                "capture_loss_lithium_kt",
                "eol_capture_loss_lithium_kt",
                "manufacturing_capture_loss_lithium_kt",
                "technology_recovery_loss_kt",
                "trade_policy_loss_kt",
                "economic_selection_loss_kt",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
