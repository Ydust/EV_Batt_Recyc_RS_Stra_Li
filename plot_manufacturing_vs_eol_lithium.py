from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
MANUFACTURING_SCRAP_FILE = ROOT / "manufacturing_scrap_mass.csv"
EOL_SCRAP_FILE = ROOT / "Scenario result" / "EV_battery_inuse_scrap.csv"
METAL_CONTENT_FILE = ROOT / "cost" / "Metal content.csv"
FIGURE_DIR = ROOT / "Figure_data"
OUTPUT_DATA = FIGURE_DIR / "manufacturing_vs_eol_lithium.csv"
OUTPUT_FIGURE = FIGURE_DIR / "manufacturing_vs_eol_lithium.svg"


def load_li_content():
    metal = pd.read_csv(METAL_CONTENT_FILE)
    metal = metal.dropna(subset=["Type", "Li"]).copy()
    metal["Li"] = pd.to_numeric(metal["Li"], errors="coerce")
    return metal.dropna(subset=["Li"]).set_index("Type")["Li"]


def eol_lithium_by_year(li_content):
    scrap = pd.read_csv(EOL_SCRAP_FILE)
    scrap["scrap"] = pd.to_numeric(scrap["scrap"], errors="coerce").fillna(0.0)
    scrap = scrap.merge(li_content.rename("li_content"), left_on="type", right_index=True, how="left")
    scrap["li_content"] = scrap["li_content"].fillna(0.0)
    scrap["eol_scrap_lithium_t"] = scrap["scrap"] * scrap["li_content"]
    return (
        scrap.groupby("Year", as_index=False)
        .agg(
            eol_scrap_t=("scrap", "sum"),
            eol_scrap_lithium_t=("eol_scrap_lithium_t", "sum"),
        )
        .rename(columns={"Year": "year"})
    )


def annual_eol_weighted_li_content(eol, default_li_content):
    weighted = eol.copy()
    weighted["weighted_li_content"] = (
        weighted["eol_scrap_lithium_t"] / weighted["eol_scrap_t"]
    )
    weighted.loc[weighted["eol_scrap_t"] <= 0, "weighted_li_content"] = default_li_content
    return weighted[["year", "weighted_li_content"]]


def manufacturing_lithium_by_year(weighted_li_content):
    manufacturing = pd.read_csv(MANUFACTURING_SCRAP_FILE)
    manufacturing["gross_manufacturing_scrap_t"] = pd.to_numeric(
        manufacturing["gross_manufacturing_scrap_t"], errors="coerce"
    ).fillna(0.0)
    manufacturing = (
        manufacturing.groupby("year", as_index=False)["gross_manufacturing_scrap_t"]
        .sum()
        .merge(weighted_li_content, on="year", how="left")
    )
    manufacturing["weighted_li_content"] = manufacturing["weighted_li_content"].ffill().bfill()
    manufacturing["manufacturing_scrap_lithium_t"] = (
        manufacturing["gross_manufacturing_scrap_t"] * manufacturing["weighted_li_content"]
    )
    return manufacturing[
        ["year", "gross_manufacturing_scrap_t", "weighted_li_content", "manufacturing_scrap_lithium_t"]
    ]


def build_plot_data():
    li_content = load_li_content()
    default_li_content = float(li_content.mean())
    eol = eol_lithium_by_year(li_content)
    weighted_li = annual_eol_weighted_li_content(eol, default_li_content)
    manufacturing = manufacturing_lithium_by_year(weighted_li)
    data = manufacturing.merge(eol, on="year", how="outer").sort_values("year")
    for column in [
        "gross_manufacturing_scrap_t",
        "manufacturing_scrap_lithium_t",
        "eol_scrap_t",
        "eol_scrap_lithium_t",
    ]:
        data[column] = data[column].fillna(0.0)
    return data


def plot(data):
    width = 980
    height = 560
    left = 86
    right = 34
    top = 62
    bottom = 76
    plot_width = width - left - right
    plot_height = height - top - bottom

    years = data["year"].astype(float)
    manufacturing = data["manufacturing_scrap_lithium_t"].astype(float) / 1000
    eol = data["eol_scrap_lithium_t"].astype(float) / 1000
    x_min = int(years.min())
    x_max = int(years.max())
    y_max = max(float(manufacturing.max()), float(eol.max()), 1.0)
    y_max = round(y_max * 1.12 + 0.5, 1)

    def x_scale(year):
        return left + (float(year) - x_min) / (x_max - x_min) * plot_width

    def y_scale(value):
        return top + plot_height - float(value) / y_max * plot_height

    def polyline(values):
        return " ".join(
            f"{x_scale(year):.1f},{y_scale(value):.1f}"
            for year, value in zip(years, values)
        )

    y_ticks = [i * y_max / 5 for i in range(6)]
    x_ticks = [2010, 2020, 2030, 2040, 2050]

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<text x="86" y="34" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#263238">Lithium in Manufacturing Scrap vs. EOL Battery Scrap</text>',
    ]

    for tick in y_ticks:
        y = y_scale(tick)
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#D6D8DC" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{left - 12}" y="{y + 4:.1f}" font-family="Arial, sans-serif" font-size="12" fill="#5F6B73" text-anchor="end">{tick:.0f}</text>'
        )

    for tick in x_ticks:
        x = x_scale(tick)
        elements.append(
            f'<line x1="{x:.1f}" y1="{top + plot_height}" x2="{x:.1f}" y2="{top + plot_height + 6}" stroke="#455A64" stroke-width="1"/>'
        )
        elements.append(
            f'<text x="{x:.1f}" y="{top + plot_height + 26}" font-family="Arial, sans-serif" font-size="12" fill="#455A64" text-anchor="middle">{tick}</text>'
        )

    elements.extend(
        [
            f'<line x1="{left}" y1="{top + plot_height}" x2="{width - right}" y2="{top + plot_height}" stroke="#455A64" stroke-width="1.2"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#455A64" stroke-width="1.2"/>',
            f'<polyline points="{polyline(manufacturing)}" fill="none" stroke="#2A9D8F" stroke-width="3.2" stroke-linejoin="round" stroke-linecap="round"/>',
            f'<polyline points="{polyline(eol)}" fill="none" stroke="#D1495B" stroke-width="3.2" stroke-linejoin="round" stroke-linecap="round"/>',
            '<text x="455" y="538" font-family="Arial, sans-serif" font-size="14" fill="#263238">Year</text>',
            '<text x="22" y="318" font-family="Arial, sans-serif" font-size="14" fill="#263238" transform="rotate(-90 22,318)">Lithium content (thousand tonnes Li)</text>',
            '<line x1="654" y1="78" x2="704" y2="78" stroke="#2A9D8F" stroke-width="3.2" stroke-linecap="round"/>',
            '<text x="714" y="82" font-family="Arial, sans-serif" font-size="13" fill="#263238">Manufacturing scrap lithium</text>',
            '<line x1="654" y1="102" x2="704" y2="102" stroke="#D1495B" stroke-width="3.2" stroke-linecap="round"/>',
            '<text x="714" y="106" font-family="Arial, sans-serif" font-size="13" fill="#263238">EOL scrap lithium</text>',
            "</svg>",
        ]
    )
    OUTPUT_FIGURE.write_text("\n".join(elements), encoding="utf-8")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = build_plot_data()
    data.to_csv(OUTPUT_DATA, index=False)
    plot(data)
    print(f"Wrote {OUTPUT_DATA}")
    print(f"Wrote {OUTPUT_FIGURE}")
    print(
        data[data["year"].isin([2023, 2030, 2040, 2050])][
            ["year", "manufacturing_scrap_lithium_t", "eol_scrap_lithium_t"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
