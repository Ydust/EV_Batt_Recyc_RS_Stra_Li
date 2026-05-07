from pathlib import Path

import analyze_direct_entry_cost as direct_metrics
import run_policy_objective_technology_elasticity as elasticity


ROOT = Path(__file__).resolve().parent
EU_TARGETS = [
    "Germany",
    "France",
    "Poland",
    "Spain",
    "Sweden",
    "Netherlands",
    "Italy",
]


def main():
    elasticity.run(
        years=[2030, 2040, 2050],
        policies=[
            "reference_policy",
            "current_policy",
            "strict_policy",
            "critical_route_policy",
        ],
        target_shares=[0.0, 0.75, 0.9, 0.95, 0.99, 1.0],
        target_mode="eu-countries",
        target_names=EU_TARGETS,
        output_suffix="eu_country_split",
        mc_draws=1,
        random_seed=20260507,
        cost_noise_sd=0.0,
        recovery_noise_sd=0.0,
        solver_methods=["highs", "highs-ds", "highs-ipm"],
    )
    summary, mix = direct_metrics.read_source_tables(["eu_country_split"])
    summary, mix = direct_metrics.normalize_tables(summary, mix)
    output_dir = (
        ROOT
        / "Figure_data"
        / "joint_policy_technology"
        / "direct_entry_cost"
        / "eu_country_split"
    )
    for output in direct_metrics.write_outputs(
        summary,
        mix,
        output_dir,
        direct_metrics.ENTRY_THRESHOLDS_PCT,
    ):
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
