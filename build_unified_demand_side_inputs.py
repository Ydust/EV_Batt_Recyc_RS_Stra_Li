from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
EOL_COLLECTED = (
    ROOT
    / "Scenario result"
    / "recycling_rate"
    / "EV_battery_inuse_scrap_collected_by_scenario.csv"
)
MANUFACTURING = ROOT / "Scenario result" / "EV_battery_manufacturing_scrap.csv"
OUT_DIR = ROOT / "unified_policy_run" / "data" / "demand_side_inputs"


OUTPUT_COLUMNS = [
    "Year",
    "region",
    "type",
    "scenario",
    "collection_rate",
    "scrap_original",
    "scrap",
    "scrap_uncollected",
    "inuse",
    "f_ebp",
    "tau_w",
    "f_tau",
    "source",
]


def read_eol_collected():
    if not EOL_COLLECTED.exists():
        raise FileNotFoundError(f"Missing collected EOL scrap input: {EOL_COLLECTED}")
    eol = pd.read_csv(EOL_COLLECTED)
    eol = eol.copy()
    eol["source"] = "eol_scrap"
    for col in OUTPUT_COLUMNS:
        if col not in eol.columns:
            eol[col] = pd.NA
    return eol[OUTPUT_COLUMNS]


def read_manufacturing_for_scenarios(scenarios):
    if not MANUFACTURING.exists():
        raise FileNotFoundError(f"Missing manufacturing scrap input: {MANUFACTURING}")
    manufacturing = pd.read_csv(MANUFACTURING)
    rows = []
    for scenario in scenarios:
        mfg = manufacturing.copy()
        mfg["scenario"] = scenario
        mfg["collection_rate"] = 1.0
        mfg["scrap_original"] = mfg["scrap"]
        mfg["scrap_uncollected"] = 0.0
        mfg["source"] = "manufacturing_scrap"
        for col in OUTPUT_COLUMNS:
            if col not in mfg.columns:
                mfg[col] = pd.NA
        rows.append(mfg[OUTPUT_COLUMNS])
    return pd.concat(rows, ignore_index=True)


def write_summary(df, path):
    summary = (
        df.groupby(["source", "scenario", "Year"], as_index=False)["scrap"]
        .sum()
        .sort_values(["source", "scenario", "Year"])
    )
    summary.to_csv(path, index=False)
    return path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    eol = read_eol_collected()
    scenarios = sorted(eol["scenario"].dropna().unique())
    manufacturing = read_manufacturing_for_scenarios(scenarios)

    eol_only = OUT_DIR / "eol_collected_by_scenario.csv"
    eol_plus_mfg = OUT_DIR / "eol_plus_manufacturing_collected_by_scenario.csv"
    summary_path = OUT_DIR / "demand_side_input_summary.csv"

    eol.to_csv(eol_only, index=False)
    combined = pd.concat([eol, manufacturing], ignore_index=True)
    combined.to_csv(eol_plus_mfg, index=False)
    write_summary(combined, summary_path)

    for path in [eol_only, eol_plus_mfg, summary_path]:
        print(path)


if __name__ == "__main__":
    main()
