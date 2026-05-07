from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
METAL_CONTENT_FILE = ROOT / "cost" / "Metal content.csv"
EOL_SCRAP_FILE = ROOT / "Scenario result" / "EV_battery_inuse_scrap.csv"
MANUFACTURING_SCRAP_FILE = ROOT / "Scenario result" / "EV_battery_manufacturing_scrap.csv"
FIGURE_DIR = ROOT / "Figure_data"
OUTPUT_DATA = FIGURE_DIR / "typical_country_lithium_scrap_trends.csv"
OUTPUT_FIGURE = FIGURE_DIR / "typical_country_lithium_scrap_trends.svg"
OUTPUT_DATA_TO_2030 = FIGURE_DIR / "typical_country_lithium_scrap_trends_to_2030.csv"
OUTPUT_FIGURE_TO_2030 = FIGURE_DIR / "typical_country_lithium_scrap_trends_to_2030.svg"

COUNTRIES = ["China", "USA", "Germany", "Korea", "Japan"]


def load_li_content():
    metal = pd.read_csv(METAL_CONTENT_FILE)
    metal = metal.dropna(subset=["Type", "Li"]).copy()
    metal["Li"] = pd.to_numeric(metal["Li"], errors="coerce")
    return metal.dropna(subset=["Li"]).set_index("Type")["Li"]


def lithium_by_country_year(path, source_name, value_col):
    li_content = load_li_content()
    data = pd.read_csv(path)
    data = data[data["region"].isin(COUNTRIES)].copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce").fillna(0.0)
    data = data.merge(li_content.rename("li_content"), left_on="type", right_index=True, how="left")
    data["li_content"] = data["li_content"].fillna(0.0)
    data[source_name] = data[value_col] * data["li_content"]
    return data.groupby(["Year", "region"], as_index=False)[source_name].sum()


def build_data():
    eol = lithium_by_country_year(EOL_SCRAP_FILE, "eol_lithium_t", "scrap")
    manufacturing = lithium_by_country_year(
        MANUFACTURING_SCRAP_FILE, "manufacturing_lithium_t", "scrap"
    )
    data = eol.merge(manufacturing, on=["Year", "region"], how="outer").fillna(0.0)
    data = data.rename(columns={"Year": "year", "region": "country"})
    return data.sort_values(["country", "year"])


def scale_points(country_data, value_col, x_scale, y_scale):
    return " ".join(
        f"{x_scale(row.year):.1f},{y_scale(getattr(row, value_col)):.1f}"
        for row in country_data.itertuples(index=False)
    )


def write_svg(data, output_figure=OUTPUT_FIGURE, title_suffix=""):
    width = 1180
    height = 760
    margin_x = 70
    margin_top = 70
    panel_w = 330
    panel_h = 250
    gutter_x = 42
    gutter_y = 70
    x_min = int(data["year"].min())
    x_max = int(data["year"].max())

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        f'<text x="70" y="38" font-family="Arial, sans-serif" font-size="21" font-weight="700" fill="#263238">Lithium in Manufacturing Scrap vs. EOL Scrap by Country{title_suffix}</text>',
        '<line x1="760" y1="34" x2="810" y2="34" stroke="#2A9D8F" stroke-width="3" stroke-linecap="round"/>',
        '<text x="820" y="38" font-family="Arial, sans-serif" font-size="13" fill="#263238">Manufacturing scrap Li</text>',
        '<line x1="760" y1="56" x2="810" y2="56" stroke="#D1495B" stroke-width="3" stroke-linecap="round"/>',
        '<text x="820" y="60" font-family="Arial, sans-serif" font-size="13" fill="#263238">EOL scrap Li</text>',
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
            float(country_data["manufacturing_lithium_t"].max()),
            float(country_data["eol_lithium_t"].max()),
            1.0,
        )
        y_max *= 1.12

        def x_scale(year):
            return plot_left + (float(year) - x_min) / (x_max - x_min) * plot_w

        def y_scale(value):
            return plot_top + plot_h - float(value) / y_max * plot_h

        elements.append(
            f'<text x="{left}" y="{top + 10}" font-family="Arial, sans-serif" font-size="16" font-weight="700" fill="#263238">{country}</text>'
        )
        for t in [0, y_max / 2, y_max]:
            y = y_scale(t)
            elements.append(
                f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_left + plot_w}" y2="{y:.1f}" stroke="#D6D8DC" stroke-width="0.8"/>'
            )
            elements.append(
                f'<text x="{plot_left - 8}" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="10" fill="#5F6B73" text-anchor="end">{t/1000:.0f}</text>'
            )
        x_tick_candidates = [2015, 2020, 2025, 2030, 2040, 2050]
        for year in [tick for tick in x_tick_candidates if x_min <= tick <= x_max]:
            x = x_scale(year)
            elements.append(
                f'<text x="{x:.1f}" y="{plot_top + plot_h + 20}" font-family="Arial, sans-serif" font-size="10" fill="#455A64" text-anchor="middle">{year}</text>'
            )
        elements.extend(
            [
                f'<line x1="{plot_left}" y1="{plot_top + plot_h}" x2="{plot_left + plot_w}" y2="{plot_top + plot_h}" stroke="#455A64" stroke-width="1"/>',
                f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_top + plot_h}" stroke="#455A64" stroke-width="1"/>',
                f'<polyline points="{scale_points(country_data, "manufacturing_lithium_t", x_scale, y_scale)}" fill="none" stroke="#2A9D8F" stroke-width="2.3" stroke-linejoin="round" stroke-linecap="round"/>',
                f'<polyline points="{scale_points(country_data, "eol_lithium_t", x_scale, y_scale)}" fill="none" stroke="#D1495B" stroke-width="2.3" stroke-linejoin="round" stroke-linecap="round"/>',
            ]
        )

    elements.append(
        '<text x="70" y="728" font-family="Arial, sans-serif" font-size="12" fill="#5F6B73">Y-axis labels are thousand tonnes Li. Manufacturing scrap is allocated only to battery-producing countries.</text>'
    )
    elements.append("</svg>")
    output_figure.write_text("\n".join(elements), encoding="utf-8")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = build_data()
    data.to_csv(OUTPUT_DATA, index=False)
    write_svg(data)
    data_to_2030 = data[data["year"] <= 2030].copy()
    data_to_2030.to_csv(OUTPUT_DATA_TO_2030, index=False)
    write_svg(data_to_2030, OUTPUT_FIGURE_TO_2030, " to 2030")
    print(f"Wrote {OUTPUT_DATA}")
    print(f"Wrote {OUTPUT_FIGURE}")
    print(f"Wrote {OUTPUT_DATA_TO_2030}")
    print(f"Wrote {OUTPUT_FIGURE_TO_2030}")
    print(
        data[data["year"].isin([2023, 2030, 2040, 2050])]
        .sort_values(["country", "year"])
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
