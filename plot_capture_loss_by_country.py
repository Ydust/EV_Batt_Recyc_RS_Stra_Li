from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


ROOT = Path(__file__).resolve().parent
INPUT_FILE = (
    ROOT
    / "Figure_data"
    / "joint_policy_technology"
    / "capture_loss_by_country_high_collection_2030_2040_2050.csv"
)
OUTPUT_DIR = ROOT / "Figure_data" / "joint_policy_technology"


TRANSLATION = {
    "China": "中国",
    "Pakistan": "巴基斯坦",
    "Nigeria": "尼日利亚",
    "Philippines": "菲律宾",
    "India": "印度",
    "Uganda": "乌干达",
    "Egypt": "埃及",
    "Kenya": "肯尼亚",
    "Mexico": "墨西哥",
    "Brazil": "巴西",
    "Sri Lanka": "斯里兰卡",
    "Myanmar": "缅甸",
    "Senegal": "塞内加尔",
    "Saudi Arabia": "沙特阿拉伯",
    "United Rep. of Tanzania": "坦桑尼亚",
    "Indonesia": "印度尼西亚",
    "Benin": "贝宁",
    "Uzbekistan": "乌兹别克斯坦",
    "USA": "美国",
    "United States": "美国",
    "Lao People's Dem. Rep.": "老挝",
    "Germany": "德国",
    "Italy": "意大利",
    "France": "法国",
    "United Kingdom": "英国",
    "Spain": "西班牙",
    "Canada": "加拿大",
    "Poland": "波兰",
    "Korea": "韩国",
    "Japan": "日本",
    "Sweden": "瑞典",
    "Norway": "挪威",
    "Hungary": "匈牙利",
    "Switzerland": "瑞士",
    "Thailand": "泰国",
    "Russian Federation": "俄罗斯",
    "Netherlands": "荷兰",
    "Australia": "澳大利亚",
    "Turkey": "土耳其",
    "Belgium": "比利时",
    "Finland": "芬兰",
    "Serbia": "塞尔维亚",
    "Czechia": "捷克",
    "Slovakia": "斯洛伐克",
    "Romania": "罗马尼亚",
    "Viet Nam": "越南",
    "Bulgaria": "保加利亚",
}


MAIN_COUNTRIES = ["China"]
INSET1_COUNTRIES = ["Korea", "United States", "Brazil", "India"]
INSET2_COUNTRIES = [
    "Japan",
    "Sweden",
    "Norway",
    "Germany",
    "United Kingdom",
    "Hungary",
    "Canada",
    "Spain",
    "Italy",
    "France",
    "Switzerland",
    "Poland",
]
INSET3_COUNTRIES = [
    "Thailand",
    "Russian Federation",
    "Netherlands",
    "Australia",
    "Turkey",
    "Belgium",
    "Finland",
    "Viet Nam",
    "Serbia",
    "Czechia",
    "Slovakia",
    "Romania",
]


def country_label(country):
    return TRANSLATION.get(country, country)


def prepare_data():
    data = pd.read_csv(INPUT_FILE)
    data.loc[data["country"] == "USA", "country"] = "United States"
    data["capture_loss_kt_li"] = data["capture_loss_lithium_t"] / 1000.0

    countries = MAIN_COUNTRIES + INSET1_COUNTRIES + INSET2_COUNTRIES + INSET3_COUNTRIES
    plot_data = data[data["country"].isin(countries)].copy()
    plot_data["country"] = pd.Categorical(
        plot_data["country"], categories=countries, ordered=True
    )
    return plot_data.sort_values(["country", "Year"]), countries


def draw_grouped_bars(
    ax,
    pivot,
    countries,
    colors,
    width,
    gap,
    ylim,
    yticks,
    show_ylabel=False,
    label_countries=None,
    tick_labelsize=12,
    country_fontsize=12,
):
    x = np.arange(len(countries))
    years = [2030, 2040, 2050]
    offsets = [-(width + gap), 0, width + gap]
    bars_for_labels = None

    for year, offset, color in zip(years, offsets, colors):
        values = pivot.reindex(countries)[year].fillna(0).to_numpy()
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            color=color,
            label=f"{year}年捕获损失",
        )
        if bars_for_labels is None:
            bars_for_labels = bars

    for i in range(1, len(countries)):
        ax.axvline(x=i - 0.5, color="lightgray", linestyle="--", linewidth=1)

    ax.set_xlim(-0.6, len(countries) - 0.4)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_xticks(x)
    ax.set_xticklabels([country_label(country) for country in countries])
    ax.tick_params(axis="x", length=0)
    ax.get_xaxis().set_visible(False)
    ax.tick_params(axis="y", direction="in", labelsize=tick_labelsize)
    ax.grid(False)
    if show_ylabel:
        ax.set_ylabel("单位：kt Li", fontsize=12)

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.0)

    y_top = ax.get_ylim()[1]
    label_set = set(countries if label_countries is None else label_countries)
    for bar, country in zip(bars_for_labels, countries):
        if country not in label_set:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_top,
            country_label(country),
            ha="center",
            va="top",
            rotation=90,
            fontsize=country_fontsize,
        )


def main():
    font_path = Path("C:/Windows/Fonts/simhei.ttf")
    if font_path.exists():
        font = FontProperties(fname=str(font_path))
        plt.rcParams["font.sans-serif"] = [font.get_name(), "DejaVu Sans"]
    else:
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False

    data, countries = prepare_data()
    pivot = data.pivot_table(
        index="country",
        columns="Year",
        values="capture_loss_kt_li",
        aggfunc="sum",
        fill_value=0,
    )

    colors = ["#B8DBB3", "#E29135", "#94C6CD"]
    width = 0.23
    gap = 0.02

    fig, ax = plt.subplots(figsize=(16, 6.2), dpi=300)
    draw_grouped_bars(
        ax,
        pivot,
        countries,
        colors,
        width,
        gap,
        ylim=(0, 16),
        yticks=np.arange(0, 17, 4),
        show_ylabel=True,
        label_countries=MAIN_COUNTRIES,
        tick_labelsize=12,
        country_fontsize=12,
    )
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.58, -0.18),
        ncol=3,
        prop={"size": 11},
        frameon=True,
    )

    ax_inset1 = fig.add_axes([0.135, 0.32, 0.17, 0.50])
    ax_inset1.set_facecolor("white")
    draw_grouped_bars(
        ax_inset1,
        pivot,
        INSET1_COUNTRIES,
        colors,
        width,
        gap,
        ylim=(0, 9),
        yticks=np.arange(0, 10, 2),
        tick_labelsize=10,
        country_fontsize=9,
    )

    ax_inset2 = fig.add_axes([0.325, 0.32, 0.30, 0.50])
    ax_inset2.set_facecolor("white")
    draw_grouped_bars(
        ax_inset2,
        pivot,
        INSET2_COUNTRIES,
        colors,
        width,
        gap,
        ylim=(0, 0.7),
        yticks=np.arange(0, 0.71, 0.1),
        tick_labelsize=9,
        country_fontsize=6,
    )

    ax_inset3 = fig.add_axes([0.66, 0.32, 0.30, 0.50])
    ax_inset3.set_facecolor("white")
    draw_grouped_bars(
        ax_inset3,
        pivot,
        INSET3_COUNTRIES,
        colors,
        width,
        gap,
        ylim=(0, 0.3),
        yticks=np.arange(0, 0.301, 0.1),
        tick_labelsize=9,
        country_fontsize=6,
    )

    fig.subplots_adjust(left=0.055, right=0.99, top=0.96, bottom=0.18)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "capture_loss_by_country_high_collection_2030_2040_2050.png"
    pdf_path = OUTPUT_DIR / "capture_loss_by_country_high_collection_2030_2040_2050.pdf"
    fig.savefig(png_path, format="png", dpi=320, transparent=True)
    fig.savefig(pdf_path, format="pdf", transparent=True)
    plt.close(fig)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
