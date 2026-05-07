from pathlib import Path
import argparse

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_DIRECT_DIR = ROOT / "Figure_data" / "joint_policy_technology" / "direct_entry_cost"
DEFAULT_MECH_DIR = ROOT / "Figure_data" / "joint_policy_technology" / "policy_technology_mechanisms"
DEFAULT_OUTPUT_DIR = ROOT / "Figure_data" / "joint_policy_technology" / "temporal_policy_technology"


def read_csv(path):
    data = pd.read_csv(path)
    if "year" in data.columns:
        data["year"] = pd.to_numeric(data["year"], errors="coerce").astype("Int64")
    return data


def to_numeric(data, columns):
    for column in columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def wide_temporal(data, index_cols, value_cols):
    frames = []
    for value_col in value_cols:
        if value_col not in data.columns:
            continue
        wide = data.pivot_table(
            index=index_cols,
            columns="year",
            values=value_col,
            aggfunc="first",
        ).reset_index()
        wide.columns = [
            f"{value_col}_{col}" if isinstance(col, (int, np.integer)) else col
            for col in wide.columns
        ]
        for start, end in [(2030, 2040), (2040, 2050), (2030, 2050)]:
            start_col = f"{value_col}_{start}"
            end_col = f"{value_col}_{end}"
            if start_col in wide.columns and end_col in wide.columns:
                wide[f"{value_col}_delta_{start}_{end}"] = wide[end_col] - wide[start_col]
                wide[f"{value_col}_ratio_{start}_{end}"] = np.where(
                    wide[start_col].abs() > 1e-12,
                    wide[end_col] / wide[start_col],
                    np.nan,
                )
                years = end - start
                wide[f"{value_col}_cagr_{start}_{end}_pct"] = np.where(
                    (wide[start_col] > 0) & (wide[end_col] > 0),
                    ((wide[end_col] / wide[start_col]) ** (1 / years) - 1) * 100,
                    np.nan,
                )
        frames.append(wide)
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on=index_cols, how="outer")
    return out


def direct_threshold_temporal(direct_dir):
    data = read_csv(direct_dir / "direct_entry_threshold_delta_vs_reference.csv")
    data = to_numeric(
        data,
        [
            "direct_share_threshold_pct",
            "direct_entry_target_share_pct",
            "entry_target_share_pct_point_delta_vs_reference",
            "direct_share_at_95pct_target_pct",
            "direct_share_at_100pct_target_pct",
        ],
    )
    return wide_temporal(
        data,
        ["draw_id", "policy_scenario", "target_region", "scope", "direct_share_threshold_pct"],
        [
            "direct_entry_target_share_pct",
            "entry_target_share_pct_point_delta_vs_reference",
            "direct_share_at_95pct_target_pct",
            "direct_share_at_100pct_target_pct",
        ],
    )


def cost_temporal(direct_dir):
    data = read_csv(direct_dir / "cost_95_to_100_access.csv")
    data = to_numeric(
        data,
        [
            "cost_delta_95_to_100",
            "cost_delta_pct",
            "cost_delta_95_to_100_vs_reference",
            "cost_per_additional_actual_target_li_t",
            "direct_technology_share_pct_target_region_100",
            "target_li_at_100",
            "global_li_at_100",
        ],
    )
    return wide_temporal(
        data,
        ["draw_id", "policy_scenario", "target_region"],
        [
            "cost_delta_95_to_100",
            "cost_delta_pct",
            "cost_delta_95_to_100_vs_reference",
            "cost_per_additional_actual_target_li_t",
            "direct_technology_share_pct_target_region_100",
            "target_li_at_100",
            "global_li_at_100",
        ],
    )


def accessible_li_temporal(mech_dir):
    data = read_csv(mech_dir / "technology_accessible_li_only_method.csv")
    data = to_numeric(
        data,
        [
            "max_accessible_recovered_lithium_t",
            "potential_recovery_pct",
            "max_accessible_li_delta_vs_reference_t",
            "max_accessible_li_delta_vs_reference_pct",
            "route_modeled_cost",
        ],
    )
    return wide_temporal(
        data,
        ["policy_scenario", "technology_allowed"],
        [
            "max_accessible_recovered_lithium_t",
            "potential_recovery_pct",
            "max_accessible_li_delta_vs_reference_t",
            "route_modeled_cost",
        ],
    )


def route_temporal(mech_dir):
    data = read_csv(mech_dir / "route_technology_switch_summary.csv")
    data = to_numeric(
        data,
        [
            "switched_route_count",
            "disappeared_route_count",
            "new_route_count",
            "net_recovered_li_delta_t",
        ],
    )
    return wide_temporal(
        data,
        ["policy_scenario"],
        [
            "switched_route_count",
            "disappeared_route_count",
            "new_route_count",
            "net_recovered_li_delta_t",
        ],
    )


def destination_temporal(mech_dir):
    data = read_csv(mech_dir / "destination_shift_vs_reference.csv")
    data = to_numeric(
        data,
        [
            "destination_recovered_lithium_t",
            "destination_recovered_li_delta_vs_reference_t",
            "Direct",
            "Hydro",
            "Pyro",
        ],
    )
    return wide_temporal(
        data,
        ["policy_scenario", "destination_iso3"],
        [
            "destination_recovered_lithium_t",
            "destination_recovered_li_delta_vs_reference_t",
            "Direct",
            "Hydro",
            "Pyro",
        ],
    )


def policy_time_summary(cost_time, route_time):
    rows = []
    for _, row in cost_time.iterrows():
        rows.append(
            {
                "indicator": "cost_delta_95_to_100",
                "policy_scenario": row["policy_scenario"],
                "target_region": row["target_region"],
                "value_2030": row.get("cost_delta_95_to_100_2030", np.nan),
                "value_2050": row.get("cost_delta_95_to_100_2050", np.nan),
                "delta_2030_2050": row.get("cost_delta_95_to_100_delta_2030_2050", np.nan),
                "ratio_2030_2050": row.get("cost_delta_95_to_100_ratio_2030_2050", np.nan),
                "cagr_2030_2050_pct": row.get("cost_delta_95_to_100_cagr_2030_2050_pct", np.nan),
            }
        )
    for _, row in route_time.iterrows():
        rows.append(
            {
                "indicator": "new_route_count",
                "policy_scenario": row["policy_scenario"],
                "target_region": "Global routes",
                "value_2030": row.get("new_route_count_2030", np.nan),
                "value_2050": row.get("new_route_count_2050", np.nan),
                "delta_2030_2050": row.get("new_route_count_delta_2030_2050", np.nan),
                "ratio_2030_2050": row.get("new_route_count_ratio_2030_2050", np.nan),
                "cagr_2030_2050_pct": row.get("new_route_count_cagr_2030_2050_pct", np.nan),
            }
        )
        rows.append(
            {
                "indicator": "net_recovered_li_delta_vs_reference",
                "policy_scenario": row["policy_scenario"],
                "target_region": "Global routes",
                "value_2030": row.get("net_recovered_li_delta_t_2030", np.nan),
                "value_2050": row.get("net_recovered_li_delta_t_2050", np.nan),
                "delta_2030_2050": row.get("net_recovered_li_delta_t_delta_2030_2050", np.nan),
                "ratio_2030_2050": row.get("net_recovered_li_delta_t_ratio_2030_2050", np.nan),
                "cagr_2030_2050_pct": row.get("net_recovered_li_delta_t_cagr_2030_2050_pct", np.nan),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Build temporal comparisons from mechanism tables.")
    parser.add_argument("--direct-dir", default=str(DEFAULT_DIRECT_DIR))
    parser.add_argument("--mechanism-dir", default=str(DEFAULT_MECH_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()
    direct_dir = Path(args.direct_dir)
    mech_dir = Path(args.mechanism_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "direct_entry_threshold_temporal.csv": direct_threshold_temporal(direct_dir),
        "cost_95_to_100_temporal.csv": cost_temporal(direct_dir),
        "technology_accessible_li_temporal.csv": accessible_li_temporal(mech_dir),
        "route_reallocation_temporal.csv": route_temporal(mech_dir),
        "destination_shift_temporal.csv": destination_temporal(mech_dir),
    }
    outputs["policy_time_summary.csv"] = policy_time_summary(
        outputs["cost_95_to_100_temporal.csv"],
        outputs["route_reallocation_temporal.csv"],
    )
    for name, data in outputs.items():
        path = output_dir / name
        data.to_csv(path, index=False)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
