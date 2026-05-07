from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT = (
    ROOT
    / "Figure_data"
    / "policy_constraint_strength"
    / "strategy2_vs_strategy3_policy_transfer_comparison.csv"
)
OUTPUT = (
    ROOT
    / "Figure_data"
    / "policy_constraint_strength"
    / "strategy2_vs_strategy3_policy_transfer_comparison.png"
)


COLORS = {"Strategy 2": "#4C78A8", "Strategy 3": "#F58518"}


def main():
    data = pd.read_csv(INPUT)
    years = sorted(data["year"].unique())
    x = range(len(years))
    width = 0.36
    offsets = {"Strategy 2": -width / 2, "Strategy 3": width / 2}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for strategy in ["Strategy 2", "Strategy 3"]:
        subset = data[data["strategy"] == strategy].set_index("year").loc[years]
        xpos = [i + offsets[strategy] for i in x]
        axes[0].bar(
            xpos,
            subset["delta_cross_border"] / 1e6,
            width=width,
            color=COLORS[strategy],
            label=strategy,
        )
        axes[1].bar(
            xpos,
            subset["delta_pct"],
            width=width,
            color=COLORS[strategy],
            label=strategy,
        )

    axes[0].axhline(0, color="#111827", linewidth=1)
    axes[1].axhline(0, color="#111827", linewidth=1)
    axes[0].set_ylabel("Change in cross-border flow\nmillion tonnes")
    axes[1].set_ylabel("Change in cross-border flow\npercent")
    axes[0].set_title("Absolute response")
    axes[1].set_title("Relative response")
    for ax in axes:
        ax.set_xticks(list(x))
        ax.set_xticklabels([str(year) for year in years], rotation=30)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    fig.suptitle(
        "Transfer-strategy response to critical-route policy stress",
        fontsize=15,
        weight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300)
    plt.close(fig)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
