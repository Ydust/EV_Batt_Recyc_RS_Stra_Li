from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
METAL_CONTENT_FILE = ROOT / "cost" / "Metal content.csv"
EOL_SCRAP_FILE = ROOT / "Scenario result" / "EV_battery_inuse_scrap.csv"
COLLECTION_RATE_FILE = ROOT / "country_recycling_rate_scenarios.csv"
MANUFACTURING_SCRAP_FILE = ROOT / "Scenario result" / "EV_battery_manufacturing_scrap.csv"
MANUFACTURING_MASS_FILE = ROOT / "manufacturing_scrap_mass.csv"
FIGURE_DIR = ROOT / "Figure_data"
GLOBAL_OUTPUT = FIGURE_DIR / "capture_loss_with_manufacturing_global.csv"
COUNTRY_OUTPUT = FIGURE_DIR / "capture_loss_with_manufacturing_country.csv"
TYPICAL_OUTPUT = FIGURE_DIR / "capture_loss_typical_countries_to_2030.csv"
TYPICAL_FIGURE = FIGURE_DIR / "capture_loss_typical_countries_to_2030.svg"

COUNTRIES = ["China", "USA", "Germany", "Korea", "Japan"]
PLOT_SCENARIO = "baseline"


def load_li_content():
    metal = pd.read_csv(METAL_CONTENT_FILE)
    metal = metal.dropna(subset=["Type", "Li"]).copy()
    metal["Li"] = pd.to_numeric(metal["Li"], errors="coerce")
    return metal.dropna(subset=["Li"]).set_index("Type")["Li"]


def eol_capture_loss(li_content):
    eol = pd.read_csv(EOL_SCRAP_FILE)
    rates = pd.read_csv(COLLECTION_RATE_FILE).rename(
        columns={"country": "region", "Year": "Year"}
    )
    eol["scrap"] = pd.to_numeric(eol["scrap"], errors="coerce").fillna(0.0)
    eol = eol.merge(li_content.rename("li_content"), left_on="type", right_index=True, how="left")
    eol["li_content"] = eol["li_content"].fillna(0.0)
    eol["eol_available_lithium_t"] = eol["scrap"] * eol["li_content"]
    eol = eol.merge(
        rates[["Year", "region", "scenario", "collection_rate"]],
        on=["Year", "region"],
        how="left",
    )
    eol["collection_rate"] = pd.to_numeric(eol["collection_rate"], errors="coerce").fillna(0.0)
    eol["eol_captured_lithium_t"] = eol["eol_available_lithium_t"] * eol["collection_rate"]
    eol["eol_capture_loss_lithium_t"] = (
        eol["eol_available_lithium_t"] - eol["eol_captured_lithium_t"]
    )
    return (
        eol.groupby(["Year", "region", "scenario"], as_index=False)[
            ["eol_available_lithium_t", "eol_captured_lithium_t", "eol_capture_loss_lithium_t"]
        ]
        .sum()
        .rename(columns={"region": "country"})
    )


def manufacturing_capture_loss(li_content):
    manufacturing = pd.read_csv(MANUFACTURING_SCRAP_FILE)
    mass = pd.read_csv(MANUFACTURING_MASS_FILE)
    mass = mass[["country", "year", "gross_manufacturing_scrap_t", "uncaptured_manufacturing_scrap_t"]].copy()
    mass["manufacturing_capture_loss_rate"] = (
        pd.to_numeric(mass["uncaptured_manufacturing_scrap_t"], errors="coerce").fillna(0.0)
        / pd.to_numeric(mass["gross_manufacturing_scrap_t"], errors="coerce").replace(0, pd.NA)
    ).fillna(0.0)
    manufacturing["scrap"] = pd.to_numeric(manufacturing["scrap"], errors="coerce").fillna(0.0)
    manufacturing = manufacturing.merge(
        li_content.rename("li_content"), left_on="type", right_index=True, how="left"
    )
    manufacturing["li_content"] = manufacturing["li_content"].fillna(0.0)
    manufacturing = manufacturing.merge(
        mass[["country", "year", "manufacturing_capture_loss_rate"]],
        left_on=["region", "Year"],
        right_on=["country", "year"],
        how="left",
    )
    manufacturing["manufacturing_capture_loss_rate"] = manufacturing[
        "manufacturing_capture_loss_rate"
    ].fillna(0.0)
    manufacturing["manufacturing_available_lithium_t"] = (
        manufacturing["scrap"] * manufacturing["li_content"]
    )
    manufacturing["manufacturing_capture_loss_lithium_t"] = (
        manufacturing["manufacturing_available_lithium_t"]
        * manufacturing["manufacturing_capture_loss_rate"]
    )
    manufacturing["manufacturing_captured_lithium_t"] = (
        manufacturing["manufacturing_available_lithium_t"]
        - manufacturing["manufacturing_capture_loss_lithium_t"]
    )
    return (
        manufacturing.groupby(["Year", "region"], as_index=False)[
            [
                "manufacturing_available_lithium_t",
                "manufacturing_captured_lithium_t",
                "manufacturing_capture_loss_lithium_t",
            ]
        ]
        .sum()
        .rename(columns={"region": "country"})
    )


def combine_losses():
    li_content = load_li_content()
    eol = eol_capture_loss(li_content)
    manufacturing = manufacturing_capture_loss(li_content)
    scenarios = eol[["scenario"]].drop_duplicates()
    manufacturing = manufacturing.merge(scenarios, how="cross")
    combined = eol.merge(manufacturing, on=["Year", "country", "scenario"], how="outer").fillna(0.0)
    combined["total_available_lithium_t"] = (
        combined["eol_available_lithium_t"] + combined["manufacturing_available_lithium_t"]
    )
    combined["total_captured_lithium_t"] = (
        combined["eol_captured_lithium_t"] + combined["manufacturing_captured_lithium_t"]
    )
    combined["total_capture_loss_lithium_t"] = (
        combined["eol_capture_loss_lithium_t"]
        + combined["manufacturing_capture_loss_lithium_t"]
    )
    combined["capture_loss_rate"] = (
        combined["total_capture_loss_lithium_t"]
        / combined["total_available_lithium_t"].replace(0, pd.NA)
    ).fillna(0.0)
    return combined.sort_values(["scenario", "Year", "country"])


def global_summary(country):
    return (
        country.groupby(["Year", "scenario"], as_index=False)[
            [
                "eol_available_lithium_t",
                "eol_capture_loss_lithium_t",
                "manufacturing_available_lithium_t",
                "manufacturing_capture_loss_lithium_t",
                "total_available_lithium_t",
                "total_capture_loss_lithium_t",
            ]
        ]
        .sum()
        .assign(
            capture_loss_rate=lambda df: (
                df["total_capture_loss_lithium_t"]
                / df["total_available_lithium_t"].replace(0, pd.NA)
            ).fillna(0.0)
        )
        .sort_values(["scenario", "Year"])
    )


def points(data, value_col, x_scale, y_scale):
    return " ".join(
        f"{x_scale(row.Year):.1f},{y_scale(getattr(row, value_col)):.1f}"
        for row in data.itertuples(index=False)
    )


def write_typical_svg(data):
    width = 1180
    height = 760
    margin_x = 70
    margin_top = 70
    panel_w = 330
    panel_h = 250
    gutter_x = 42
    gutter_y = 70
    x_min = int(data["Year"].min())
    x_max = int(data["Year"].max())
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<text x="70" y="38" font-family="Arial, sans-serif" font-size="21" font-weight="700" fill="#263238">Capture Loss by Country to 2030 (baseline collection)</text>',
        '<line x1="720" y1="34" x2="770" y2="34" stroke="#D1495B" stroke-width="3" stroke-linecap="round"/>',
        '<text x="780" y="38" font-family="Arial, sans-serif" font-size="13" fill="#263238">EOL capture loss Li</text>',
        '<line x1="720" y1="56" x2="770" y2="56" stroke="#2A9D8F" stroke-width="3" stroke-linecap="round"/>',
        '<text x="780" y="60" font-family="Arial, sans-serif" font-size="13" fill="#263238">Manufacturing capture loss Li</text>',
    ]
    for idx, country in enumerate(COUNTRIES):
        country_data = data[data["country"] == country].copy()
        row = idx // 3
        col = idx % 3
        left = margin_x + col * (panel_w + gutter_x)
        top = margin_top + row * (panel_h + gutter_y)
        plot_left = left + 48
        plot_top = top + 30
        plot_w = panel_w - 62
        plot_h = panel_h - 58
        y_max = max(
            float(country_data["eol_capture_loss_lithium_t"].max()),
            float(country_data["manufacturing_capture_loss_lithium_t"].max()),
            1.0,
        ) * 1.12

        def x_scale(year):
            return plot_left + (float(year) - x_min) / (x_max - x_min) * plot_w

        def y_scale(value):
            return plot_top + plot_h - float(value) / y_max * plot_h

        elements.append(
            f'<text x="{left}" y="{top + 10}" font-family="Arial, sans-serif" font-size="16" font-weight="700" fill="#263238">{country}</text>'
        )
        for tick in [0, y_max / 2, y_max]:
            y = y_scale(tick)
            elements.append(
                f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_left + plot_w}" y2="{y:.1f}" stroke="#D6D8DC" stroke-width="0.8"/>'
            )
            elements.append(
                f'<text x="{plot_left - 8}" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="10" fill="#5F6B73" text-anchor="end">{tick/1000:.1f}</text>'
            )
        for year in [2015, 2020, 2025, 2030]:
            x = x_scale(year)
            elements.append(
                f'<text x="{x:.1f}" y="{plot_top + plot_h + 20}" font-family="Arial, sans-serif" font-size="10" fill="#455A64" text-anchor="middle">{year}</text>'
            )
        elements.extend(
            [
                f'<line x1="{plot_left}" y1="{plot_top + plot_h}" x2="{plot_left + plot_w}" y2="{plot_top + plot_h}" stroke="#455A64" stroke-width="1"/>',
                f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_top + plot_h}" stroke="#455A64" stroke-width="1"/>',
                f'<polyline points="{points(country_data, "eol_capture_loss_lithium_t", x_scale, y_scale)}" fill="none" stroke="#D1495B" stroke-width="2.3" stroke-linejoin="round" stroke-linecap="round"/>',
                f'<polyline points="{points(country_data, "manufacturing_capture_loss_lithium_t", x_scale, y_scale)}" fill="none" stroke="#2A9D8F" stroke-width="2.3" stroke-linejoin="round" stroke-linecap="round"/>',
            ]
        )
    elements.append(
        '<text x="70" y="728" font-family="Arial, sans-serif" font-size="12" fill="#5F6B73">Y-axis labels are thousand tonnes Li. Manufacturing capture loss uses country-year manufacturing scrap capture rates.</text>'
    )
    elements.append("</svg>")
    TYPICAL_FIGURE.write_text("\n".join(elements), encoding="utf-8")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    country = combine_losses()
    glob = global_summary(country)
    country.to_csv(COUNTRY_OUTPUT, index=False)
    glob.to_csv(GLOBAL_OUTPUT, index=False)
    typical = country[
        (country["scenario"] == PLOT_SCENARIO)
        & (country["country"].isin(COUNTRIES))
        & (country["Year"] <= 2030)
    ].copy()
    typical.to_csv(TYPICAL_OUTPUT, index=False)
    write_typical_svg(typical)
    print(f"Wrote {GLOBAL_OUTPUT}")
    print(f"Wrote {COUNTRY_OUTPUT}")
    print(f"Wrote {TYPICAL_OUTPUT}")
    print(f"Wrote {TYPICAL_FIGURE}")
    print(
        glob[(glob["scenario"] == PLOT_SCENARIO) & (glob["Year"].isin([2023, 2030, 2040, 2050]))][
            [
                "Year",
                "eol_capture_loss_lithium_t",
                "manufacturing_capture_loss_lithium_t",
                "total_capture_loss_lithium_t",
                "capture_loss_rate",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
