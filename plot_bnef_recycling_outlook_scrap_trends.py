from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
BNEF_RECYCLING_OUTLOOK = ROOT / "Lithium-ion Battery Recycling Market Outlook 2024.xlsm"
FIGURE_DIR = ROOT / "Figure_data"
OUTPUT_DATA = FIGURE_DIR / "bnef_recycling_outlook_scrap_trends.csv"
OUTPUT_FIGURE = FIGURE_DIR / "bnef_recycling_outlook_scrap_trends.svg"
FORECAST_END_YEAR = 2050
FORECAST_BASE_YEAR = 2030


def extend_series_to_2050(data, columns):
    data = data.copy()
    data["data_status"] = "BNEF_observed"
    last_year = int(data["year"].max())
    if last_year >= FORECAST_END_YEAR:
        return data

    base = data[data["year"] == FORECAST_BASE_YEAR]
    base_year = FORECAST_BASE_YEAR if not base.empty else int(data["year"].iloc[-2])
    last = data[data["year"] == last_year].iloc[0]
    base_row = data[data["year"] == base_year].iloc[0]
    forecast_rows = []
    for year in range(last_year + 1, FORECAST_END_YEAR + 1):
        row = {"year": year, "data_status": "linear_forecast_from_2030_2035_trend"}
        for column in columns:
            annual_delta = (float(last[column]) - float(base_row[column])) / (last_year - base_year)
            row[column] = max(0.0, float(last[column]) + annual_delta * (year - last_year))
        forecast_rows.append(row)
    return pd.concat([data, pd.DataFrame(forecast_rows)], ignore_index=True)


def load_figure2():
    raw = pd.read_excel(
        BNEF_RECYCLING_OUTLOOK,
        sheet_name="Figure 2",
        header=None,
        engine="openpyxl",
    )
    years = [int(value) for value in raw.iloc[7, 2:] if pd.notna(value)]

    def row_values(label):
        row = raw[raw.iloc[:, 1] == label]
        if row.empty:
            raise ValueError(f"Could not find row: {label}")
        return pd.to_numeric(row.iloc[0, 2 : 2 + len(years)], errors="coerce").fillna(0.0).to_numpy()

    production_scrap = row_values("Production scrap")
    total = row_values("Total")
    non_production = total - production_scrap
    data = pd.DataFrame(
        {
            "year": years,
            "manufacturing_production_scrap_gwh": production_scrap,
            "eol_and_other_recycling_availability_gwh": non_production,
            "total_recycling_availability_gwh": total,
        }
    )
    data = extend_series_to_2050(
        data,
        [
            "manufacturing_production_scrap_gwh",
            "eol_and_other_recycling_availability_gwh",
            "total_recycling_availability_gwh",
        ],
    )
    return data


def make_points(data, value_col, x_scale, y_scale):
    return " ".join(
        f"{x_scale(row.year):.1f},{y_scale(getattr(row, value_col)):.1f}"
        for row in data.itertuples(index=False)
    )


def write_svg(data):
    width = 980
    height = 560
    left = 82
    right = 34
    top = 62
    bottom = 76
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_min = int(data["year"].min())
    x_max = int(data["year"].max())
    y_max = float(
        data[
            [
                "manufacturing_production_scrap_gwh",
                "eol_and_other_recycling_availability_gwh",
            ]
        ].max().max()
    )
    y_max = max(1.0, y_max * 1.12)

    def x_scale(year):
        return left + (float(year) - x_min) / (x_max - x_min) * plot_width

    def y_scale(value):
        return top + plot_height - float(value) / y_max * plot_height

    y_ticks = [i * y_max / 5 for i in range(6)]
    x_ticks = [2015, 2020, 2025, 2030, 2035, 2040, 2045, 2050]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        '<text x="82" y="34" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#263238">BNEF Recycling Outlook: Production Scrap vs. EOL/Other Availability to 2050</text>',
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
            f'<polyline points="{make_points(data, "manufacturing_production_scrap_gwh", x_scale, y_scale)}" fill="none" stroke="#2A9D8F" stroke-width="3.2" stroke-linejoin="round" stroke-linecap="round"/>',
            f'<polyline points="{make_points(data, "eol_and_other_recycling_availability_gwh", x_scale, y_scale)}" fill="none" stroke="#D1495B" stroke-width="3.2" stroke-linejoin="round" stroke-linecap="round"/>',
            '<text x="455" y="538" font-family="Arial, sans-serif" font-size="14" fill="#263238">Year</text>',
            '<text x="22" y="330" font-family="Arial, sans-serif" font-size="14" fill="#263238" transform="rotate(-90 22,330)">Battery recycling availability (GWh)</text>',
            '<line x1="604" y1="78" x2="654" y2="78" stroke="#2A9D8F" stroke-width="3.2" stroke-linecap="round"/>',
            '<text x="664" y="82" font-family="Arial, sans-serif" font-size="13" fill="#263238">Production scrap</text>',
            '<line x1="604" y1="102" x2="654" y2="102" stroke="#D1495B" stroke-width="3.2" stroke-linecap="round"/>',
            '<text x="664" y="106" font-family="Arial, sans-serif" font-size="13" fill="#263238">EOL and other availability</text>',
            "</svg>",
        ]
    )
    OUTPUT_FIGURE.write_text("\n".join(elements), encoding="utf-8")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_figure2()
    data.to_csv(OUTPUT_DATA, index=False)
    write_svg(data)
    print(f"Wrote {OUTPUT_DATA}")
    print(f"Wrote {OUTPUT_FIGURE}")
    print(
        data[data["year"].isin([2023, 2025, 2030, 2035, 2040, 2050])][
            [
                "year",
                "manufacturing_production_scrap_gwh",
                "eol_and_other_recycling_availability_gwh",
                "data_status",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
