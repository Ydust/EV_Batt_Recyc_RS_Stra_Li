from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_figure1_concept import (
    COUNTRY_COLORS,
    OUTPUT_DIR,
    ROUTE_POLICY,
    load_source_to_destination_cascade,
)


YEARS = [2030, 2040, 2050]


def draw_panel(ax, data, year):
    stage_cols = [
        "Potential by source",
        "Captured by source",
        "Open-access by destination",
        f"{ROUTE_POLICY} by destination",
    ]
    stage_labels = [
        "Source\npotential",
        "Captured source\nsupply",
        "Unconstrained\navailability",
        "Policy-constrained\navailability",
    ]
    x = np.arange(len(stage_cols))
    bottom = np.zeros(len(stage_cols))
    for idx, row in data.iterrows():
        values = row[stage_cols].to_numpy(dtype=float)
        ax.fill_between(
            x,
            bottom,
            bottom + values,
            color=COUNTRY_COLORS[min(idx, len(COUNTRY_COLORS) - 1)],
            alpha=0.78,
            linewidth=0,
            label=row["country"] if year == YEARS[-1] else None,
        )
        ax.plot(x, bottom + values, color="white", linewidth=0.6, alpha=0.8)
        bottom += values
    totals = data[stage_cols].sum().to_numpy(dtype=float)
    ax.plot(x, totals, color="#111827", linewidth=1.2)
    ax.scatter(x, totals, s=34, color="#111827", edgecolor="white", linewidth=0.5)
    ax.set_title(str(year), fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, fontsize=7)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(totals.max() * 1.16, 1))
    for i, value in enumerate(totals):
        ax.text(i, value + totals.max() * 0.03, f"{value:,.0f}", ha="center", fontsize=7)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_data = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=220, sharey=False)
    for ax, year in zip(axes, YEARS):
        data = load_source_to_destination_cascade(year=year)
        all_data.append(data.assign(year=year))
        draw_panel(ax, data, year)
    axes[0].set_ylabel("Lithium (kt Li)")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=7.5,
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "D-panel trend: source potential to destination access",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])
    out_png = OUTPUT_DIR / "Figure1_D_panel_trend_2030_2040_2050.png"
    out_csv = OUTPUT_DIR / "Figure1_D_panel_trend_2030_2040_2050.csv"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    pd.concat(all_data, ignore_index=True).to_csv(out_csv, index=False)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
