from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_FILE = (
    ROOT / "unified_policy_run" / "figures" / "fig2_route_maps"
    / "route_access_loss_by_policy_year.csv"
)
OUT_DIR = ROOT / "unified_policy_run" / "figures" / "fig2_route_maps"

POLICIES = ["current_policy", "reference_policy", "strict_policy", "critical_route_policy"]
POLICY_LABELS = {
    "current_policy": "Current",
    "reference_policy": "Reference",
    "strict_policy": "Strict",
    "critical_route_policy": "Critical-route",
}
POLICY_COLORS = {
    "current_policy":        "#4E79A7",
    "reference_policy":      "#59A14F",
    "strict_policy":         "#E15759",
    "critical_route_policy": "#B07AA1",
}


def main():
    plt.rcParams.update({"font.family": "Arial"})
    df = pd.read_csv(DATA_FILE)

    fig, ax = plt.subplots(figsize=(9.5, 5.2), dpi=300)
    for policy in POLICIES:
        sub = df[df["policy_scenario"] == policy].sort_values("year")
        if sub.empty:
            continue
        ax.plot(
            sub["year"], sub["route_access_loss_kt_li"],
            color=POLICY_COLORS[policy],
            linewidth=2.2, marker="o", markersize=4,
            markeredgecolor="white", markeredgewidth=0.7,
            label=POLICY_LABELS[policy],
        )

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Route-access loss (kt Li)", fontsize=11)
    ax.set_title("Annual route-access loss vs the route-access-unconstrained benchmark",
                 fontsize=12, pad=10)
    ax.set_xlim(2024.5, 2050.5)
    ax.set_xticks([2025, 2030, 2035, 2040, 2045, 2050])
    ax.grid(axis="y", color="0.90", linewidth=0.6)
    ax.tick_params(axis="both", labelsize=9.5, direction="in")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=10, title="Policy", title_fontsize=10.5)

    fig.text(
        0.5, 0.005,
        "Per-year sum of route-level reductions in battery-embedded secondary Li relative to the route-access-unconstrained benchmark.",
        ha="center", fontsize=8.5, color="#374151",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / "route_access_loss_annual_by_policy.png"
    pdf = OUT_DIR / "route_access_loss_annual_by_policy.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
