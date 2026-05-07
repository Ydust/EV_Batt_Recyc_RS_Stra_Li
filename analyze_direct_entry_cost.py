import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ELASTICITY_DIR = (
    ROOT
    / "Figure_data"
    / "joint_policy_technology"
    / "policy_objective_technology_elasticity"
)
OUTPUT_DIR = (
    ROOT
    / "Figure_data"
    / "joint_policy_technology"
    / "direct_entry_cost"
)
ENTRY_THRESHOLDS_PCT = [5.0, 20.0, 50.0]
DEFAULT_SOURCE_DIRS = ["root", "aggregate_us_eu", "eu_country_split"]


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def source_path(source_dir, file_name):
    return ELASTICITY_DIR / file_name if source_dir in {"", "root"} else ELASTICITY_DIR / source_dir / file_name


def read_source_tables(source_dirs):
    summary_frames = []
    mix_frames = []
    for source_dir in source_dirs:
        summary_path = source_path(source_dir, "policy_objective_technology_elasticity_summary.csv")
        mix_path = source_path(source_dir, "policy_objective_technology_elasticity_mix.csv")
        if not summary_path.exists() or not mix_path.exists():
            print(f"Skipping missing source: {source_dir or '[root]'}")
            continue
        summary = pd.read_csv(summary_path)
        mix = pd.read_csv(mix_path)
        summary["source_run"] = "root" if source_dir in {"", "root"} else source_dir
        mix["source_run"] = "root" if source_dir in {"", "root"} else source_dir
        summary_frames.append(summary)
        mix_frames.append(mix)
    if not summary_frames or not mix_frames:
        raise FileNotFoundError("No usable elasticity summary/mix source tables were found.")
    return pd.concat(summary_frames, ignore_index=True), pd.concat(mix_frames, ignore_index=True)


def normalize_tables(summary, mix):
    for table in [summary, mix]:
        if "draw_id" not in table.columns:
            table["draw_id"] = 0
        if "random_seed" not in table.columns:
            table["random_seed"] = np.nan
        for column in ["draw_id", "year", "target_share_of_max"]:
            table[column] = pd.to_numeric(table[column], errors="coerce")
    numeric_summary = [
        "target_constraint_li_t",
        "solver_target_constraint_li_t",
        "max_target_recovered_lithium_t",
        "global_recovered_lithium_t",
        "target_recovered_lithium_t",
        "target_max_attainment_pct",
        "route_modeled_cost",
        "processed_scrap_t",
    ]
    numeric_mix = ["recovered_lithium_t", "technology_share_pct"]
    for column in numeric_summary:
        if column in summary.columns:
            summary[column] = pd.to_numeric(summary[column], errors="coerce")
    for column in numeric_mix:
        if column in mix.columns:
            mix[column] = pd.to_numeric(mix[column], errors="coerce")

    solve_status = (
        summary["solve_status"]
        if "solve_status" in summary.columns
        else pd.Series("success", index=summary.index)
    )
    summary = summary[solve_status.fillna("success") != "failed"].copy()
    summary["draw_id"] = summary["draw_id"].fillna(0).astype(int)
    mix["draw_id"] = mix["draw_id"].fillna(0).astype(int)
    summary["year"] = summary["year"].astype(int)
    mix["year"] = mix["year"].astype(int)
    return summary, mix


def dedupe_summary(summary):
    keys = ["draw_id", "year", "policy_scenario", "target_region", "target_share_of_max"]
    sort_cols = keys + ["source_run"]
    return summary.sort_values(sort_cols).drop_duplicates(keys, keep="last")


def direct_entry_thresholds(mix, thresholds_pct):
    direct = mix[mix["technology"] == "Direct"].copy()
    keys = ["draw_id", "year", "policy_scenario", "target_region", "scope"]
    rows = []
    for key_values, group in direct.groupby(keys, dropna=False):
        group = group.sort_values("target_share_of_max")
        base = dict(zip(keys, key_values))
        for threshold in thresholds_pct:
            entered = group[group["technology_share_pct"] > threshold]
            row = {
                **base,
                "direct_share_threshold_pct": threshold,
                "direct_entry_target_share": (
                    float(entered["target_share_of_max"].min()) if not entered.empty else np.nan
                ),
                "direct_entry_target_share_pct": (
                    float(entered["target_share_of_max"].min()) * 100.0
                    if not entered.empty
                    else np.nan
                ),
                "direct_share_at_entry_pct": (
                    float(
                        entered.loc[
                            entered["target_share_of_max"].idxmin(),
                            "technology_share_pct",
                        ]
                    )
                    if not entered.empty
                    else np.nan
                ),
            }
            for share in [0.95, 1.0]:
                match = group[np.isclose(group["target_share_of_max"], share)]
                row[f"direct_share_at_{int(share * 100)}pct_target_pct"] = (
                    float(match["technology_share_pct"].iloc[0]) if not match.empty else np.nan
                )
            rows.append(row)
    return pd.DataFrame(rows)


def threshold_delta_vs_reference(thresholds):
    keys = ["draw_id", "year", "target_region", "scope", "direct_share_threshold_pct"]
    reference = thresholds[thresholds["policy_scenario"] == "reference_policy"][
        keys + ["direct_entry_target_share"]
    ].rename(columns={"direct_entry_target_share": "reference_entry_target_share"})
    compared = thresholds.merge(reference, on=keys, how="left")
    compared["entry_target_share_delta_vs_reference"] = (
        compared["direct_entry_target_share"] - compared["reference_entry_target_share"]
    )
    compared["entry_target_share_pct_point_delta_vs_reference"] = (
        compared["entry_target_share_delta_vs_reference"] * 100.0
    )
    return compared


def pick_share(group, share):
    match = group[np.isclose(group["target_share_of_max"], share)]
    if match.empty:
        return None
    return match.iloc[0]


def add_direct_share_columns(cost_rows, mix):
    direct = mix[mix["technology"] == "Direct"].copy()
    direct = direct[
        [
            "draw_id",
            "year",
            "policy_scenario",
            "target_region",
            "target_share_of_max",
            "scope",
            "technology_share_pct",
            "recovered_lithium_t",
        ]
    ]
    pivot = direct.pivot_table(
        index=["draw_id", "year", "policy_scenario", "target_region", "target_share_of_max"],
        columns="scope",
        values=["technology_share_pct", "recovered_lithium_t"],
        aggfunc="first",
    )
    pivot.columns = [
        f"direct_{metric}_{scope}" for metric, scope in pivot.columns.to_flat_index()
    ]
    pivot = pivot.reset_index()
    out = cost_rows.merge(
        pivot.add_suffix("_95").rename(
            columns={
                "draw_id_95": "draw_id",
                "year_95": "year",
                "policy_scenario_95": "policy_scenario",
                "target_region_95": "target_region",
                "target_share_of_max_95": "target_share_95_join",
            }
        ),
        left_on=["draw_id", "year", "policy_scenario", "target_region", "share_95"],
        right_on=["draw_id", "year", "policy_scenario", "target_region", "target_share_95_join"],
        how="left",
    )
    out = out.merge(
        pivot.add_suffix("_100").rename(
            columns={
                "draw_id_100": "draw_id",
                "year_100": "year",
                "policy_scenario_100": "policy_scenario",
                "target_region_100": "target_region",
                "target_share_of_max_100": "target_share_100_join",
            }
        ),
        left_on=["draw_id", "year", "policy_scenario", "target_region", "share_100"],
        right_on=["draw_id", "year", "policy_scenario", "target_region", "target_share_100_join"],
        how="left",
    )
    return out.drop(columns=["target_share_95_join", "target_share_100_join"], errors="ignore")


def cost_from_95_to_100(summary, mix):
    keys = ["draw_id", "year", "policy_scenario", "target_region"]
    rows = []
    for key_values, group in summary.groupby(keys, dropna=False):
        row_95 = pick_share(group, 0.95)
        row_100 = pick_share(group, 1.0)
        if row_95 is None or row_100 is None:
            continue
        base = dict(zip(keys, key_values))
        cost_delta = row_100["route_modeled_cost"] - row_95["route_modeled_cost"]
        target_constraint_delta = (
            row_100["target_constraint_li_t"] - row_95["target_constraint_li_t"]
        )
        actual_target_li_delta = (
            row_100["target_recovered_lithium_t"] - row_95["target_recovered_lithium_t"]
        )
        global_li_delta = (
            row_100["global_recovered_lithium_t"] - row_95["global_recovered_lithium_t"]
        )
        rows.append(
            {
                **base,
                "share_95": 0.95,
                "share_100": 1.0,
                "cost_at_95": row_95["route_modeled_cost"],
                "cost_at_100": row_100["route_modeled_cost"],
                "cost_delta_95_to_100": cost_delta,
                "cost_delta_pct": (
                    cost_delta / row_95["route_modeled_cost"] * 100.0
                    if row_95["route_modeled_cost"] > 0
                    else np.nan
                ),
                "target_constraint_li_delta_t": target_constraint_delta,
                "actual_target_li_delta_t": actual_target_li_delta,
                "global_recovered_li_delta_t": global_li_delta,
                "cost_per_additional_target_constraint_li_t": (
                    cost_delta / target_constraint_delta
                    if target_constraint_delta > 0
                    else np.nan
                ),
                "cost_per_additional_actual_target_li_t": (
                    cost_delta / actual_target_li_delta
                    if actual_target_li_delta > 0
                    else np.nan
                ),
                "cost_per_additional_global_li_t": (
                    cost_delta / global_li_delta if global_li_delta > 0 else np.nan
                ),
                "target_li_at_95": row_95["target_recovered_lithium_t"],
                "target_li_at_100": row_100["target_recovered_lithium_t"],
                "global_li_at_95": row_95["global_recovered_lithium_t"],
                "global_li_at_100": row_100["global_recovered_lithium_t"],
                "target_attainment_at_95_pct": row_95["target_max_attainment_pct"],
                "target_attainment_at_100_pct": row_100["target_max_attainment_pct"],
            }
        )
    cost_rows = pd.DataFrame(rows)
    if cost_rows.empty:
        return cost_rows
    cost_rows = add_direct_share_columns(cost_rows, mix)
    reference_keys = ["draw_id", "year", "target_region"]
    reference = cost_rows[cost_rows["policy_scenario"] == "reference_policy"][
        reference_keys + ["cost_delta_95_to_100"]
    ].rename(columns={"cost_delta_95_to_100": "reference_cost_delta_95_to_100"})
    cost_rows = cost_rows.merge(reference, on=reference_keys, how="left")
    cost_rows["cost_delta_95_to_100_vs_reference"] = (
        cost_rows["cost_delta_95_to_100"] - cost_rows["reference_cost_delta_95_to_100"]
    )
    return cost_rows


def write_outputs(summary, mix, output_dir, thresholds_pct):
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = dedupe_summary(summary)
    thresholds = direct_entry_thresholds(mix, thresholds_pct)
    threshold_delta = threshold_delta_vs_reference(thresholds)
    costs = cost_from_95_to_100(summary, mix)

    thresholds_path = output_dir / "direct_entry_thresholds.csv"
    threshold_delta_path = output_dir / "direct_entry_threshold_delta_vs_reference.csv"
    costs_path = output_dir / "cost_95_to_100_access.csv"
    thresholds.to_csv(thresholds_path, index=False)
    threshold_delta.to_csv(threshold_delta_path, index=False)
    costs.to_csv(costs_path, index=False)
    return thresholds_path, threshold_delta_path, costs_path


def main():
    parser = argparse.ArgumentParser(
        description="Summarize Direct entry thresholds and 95-to-100% target-access costs."
    )
    parser.add_argument(
        "--source-dirs",
        default=",".join(DEFAULT_SOURCE_DIRS),
        help="Comma-separated subfolders under the elasticity output directory. Empty entry means root.",
    )
    parser.add_argument(
        "--entry-thresholds-pct",
        default=",".join(str(value) for value in ENTRY_THRESHOLDS_PCT),
        help="Comma-separated Direct share thresholds in percent.",
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()
    summary, mix = read_source_tables(parse_csv(args.source_dirs))
    summary, mix = normalize_tables(summary, mix)
    outputs = write_outputs(
        summary,
        mix,
        Path(args.output_dir),
        [float(value) for value in parse_csv(args.entry_thresholds_pct)],
    )
    for output in outputs:
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
