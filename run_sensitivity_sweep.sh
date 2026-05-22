#!/usr/bin/env bash
# Complete 3-level sensitivity sweep on V1 baseline.
# For each config, runs Fig 3 (5 PyroHydro scenarios, 26 years, 5 policies)
# and Fig 4 (8 mitigations, 5 policies, 3 years), then stores outputs
# under unified_policy_run/sensitivity_runs/<config>/.

set -uo pipefail

ROOT="D:/文档/EV_Batt_Recyc_RS_Stra_Li"
cd "$ROOT"

YEARS="2025,2026,2027,2028,2029,2030,2031,2032,2033,2034,2035,2036,2037,2038,2039,2040,2041,2042,2043,2044,2045,2046,2047,2048,2049,2050"
POLICIES="reference_policy,current_policy,strict_policy,critical_route_policy,open_policy"
POLICY_DIR="Figure_data/joint_policy_technology"
BASELINE_POLICY_CSV="$POLICY_DIR/waste_trade_policy_constraints_reference_relaxed.csv"
SWEEP_BASE="unified_policy_run/sensitivity_runs"

# Backup original files once
cp scenario_transport_paths.py scenario_transport_paths.py.sweep_backup
cp run_dynamic_scale_policy_technology.py run_dynamic_scale_policy_technology.py.sweep_backup
cp run_lithium_loss_scenarios_unified.py run_lithium_loss_scenarios_unified.py.sweep_backup
cp "$BASELINE_POLICY_CSV" "$BASELINE_POLICY_CSV.sweep_backup"

mkdir -p "$SWEEP_BASE"

restore_files() {
  cp scenario_transport_paths.py.sweep_backup scenario_transport_paths.py
  cp run_dynamic_scale_policy_technology.py.sweep_backup run_dynamic_scale_policy_technology.py
  cp run_lithium_loss_scenarios_unified.py.sweep_backup run_lithium_loss_scenarios_unified.py
  cp "$BASELINE_POLICY_CSV.sweep_backup" "$BASELINE_POLICY_CSV"
}

# Sets delay_cost in scenario_transport_paths.py
set_delay_cost() {
  local val="$1"
  python -c "
text = open('scenario_transport_paths.py').read()
import re
text2 = re.sub(r'DEFAULT_DELAY_COST_USD_PER_T_DAY = [\d.]+',
               'DEFAULT_DELAY_COST_USD_PER_T_DAY = ${val}', text)
open('scenario_transport_paths.py','w').write(text2)
"
}

# Sets policy_penalty for current_policy BaselParty->China
set_china_penalty() {
  local val="$1"
  python -c "
import pandas as pd
df = pd.read_csv('$BASELINE_POLICY_CSV')
mask = (df['scenario']=='current_policy') & (df['destination_country']=='China')
df.loc[mask, 'policy_penalty_usd_per_t'] = ${val}
df.to_csv('$BASELINE_POLICY_CSV', index=False)
print(f'set China penalty -> ${val}')
"
}

# Sets lithium_price_multiplier in run_lithium_loss_scenarios_unified.py
set_lithium_price_mult() {
  local val="$1"
  python -c "
import re
text = open('run_lithium_loss_scenarios_unified.py').read()
# lithium_aware_high_price block
text2 = re.sub(r'(\"lithium_aware_high_price\"[^}]*\"lithium_price_multiplier\":\s*)[\d.]+',
               r'\g<1>${val}', text, count=1)
# combined_mitigation block (also has lithium_price_multiplier)
text2 = re.sub(r'(\"combined_mitigation\"[^}]*\"lithium_price_multiplier\":\s*)[\d.]+',
               r'\g<1>${val}', text2, count=1)
open('run_lithium_loss_scenarios_unified.py','w').write(text2)
"
}

run_fig3_sweep() {
  local out_dir="$1"
  local direct_mult="$2"
  local avail_thr="$3"
  mkdir -p "$out_dir"
  export GUROBI_THREADS=4

  run_one() {
    local name=$1 pw=$2 groups=$3
    python run_dynamic_scale_policy_technology.py \
      --years "$YEARS" \
      --policies "$POLICIES" \
      --include-pyrohydro \
      --pyrohydro-pyro-weight "$pw" \
      --pyrohydro-group-multipliers "$groups" \
      --direct-cost-multiplier "$direct_mult" \
      --availability-threshold "$avail_thr" \
      --lp-solver gurobi --skip-fixed \
      --output-dir "$out_dir/pyrohydro_sensitivity_${name}_unified" \
      > "$out_dir/log_${name}.txt" 2>&1 \
      && echo "[$(date +%H:%M)] $name done" >&2 \
      || echo "[$(date +%H:%M)] $name FAILED" >&2
  }

  echo "[$(date +%H:%M)] >>> Fig3 batch 1: conservative + s2 + medium" >&2
  run_one conservative 0.35 "developed=0.85,ev_producer=0.90,other=1.00" &
  run_one s2 0.30 "developed=0.815,ev_producer=0.86,other=0.95" &
  run_one medium 0.25 "developed=0.78,ev_producer=0.82,other=0.90" &
  wait
  echo "[$(date +%H:%M)] >>> Fig3 batch 2: s4 + s5" >&2
  run_one s4 0.20 "developed=0.74,ev_producer=0.785,other=0.875" &
  run_one s5 0.15 "developed=0.70,ev_producer=0.75,other=0.85" &
  wait
}

run_fig4_sweep() {
  local out_dir="$1"
  local direct_mult="$2"
  local avail_thr="$3"
  mkdir -p "$out_dir"
  export GUROBI_THREADS=8

  echo "[$(date +%H:%M)] >>> Fig4 starting" >&2
  python run_lithium_loss_scenarios_unified.py \
    --include-pyrohydro \
    --pyrohydro-pyro-weight 0.25 \
    --pyrohydro-group-multipliers "developed=0.78,ev_producer=0.82,other=0.90" \
    --direct-cost-multiplier "$direct_mult" \
    --availability-threshold "$avail_thr" \
    --include-max-li \
    --output-dir "$out_dir" \
    > "$out_dir/log_fig4.txt" 2>&1 \
    && echo "[$(date +%H:%M)] Fig4 done" >&2 \
    || echo "[$(date +%H:%M)] Fig4 FAILED" >&2
}

run_config() {
  local name="$1"
  local direct_mult="$2"
  local avail_thr="$3"
  local fig3_dir="$SWEEP_BASE/$name/fig3_pyrohydro_sensitivity_unified"
  local fig4_dir="$SWEEP_BASE/$name/lithium_loss_scenarios_unified"
  echo "[$(date +%H:%M)] === Config: $name (direct_mult=$direct_mult, avail=$avail_thr) ===" >&2
  run_fig3_sweep "$fig3_dir" "$direct_mult" "$avail_thr"
  run_fig4_sweep "$fig4_dir" "$direct_mult" "$avail_thr"
  echo "[$(date +%H:%M)] === Config $name DONE ===" >&2
}

run_config_fig4_only() {
  local name="$1"
  local fig4_dir="$SWEEP_BASE/$name/lithium_loss_scenarios_unified"
  echo "[$(date +%H:%M)] === Config (Fig4 only): $name ===" >&2
  run_fig4_sweep "$fig4_dir" 1.2 0.6
  echo "[$(date +%H:%M)] === Config $name DONE ===" >&2
}

# ==========================================
# RUN ALL CONFIGS
# ==========================================
# direct_cost_multiplier sweep
echo "[$(date +%H:%M)] ## SWEEP 1: direct_cost_multiplier" >&2
restore_files
run_config "direct_mult_1.0" 1.0 0.6

restore_files
run_config "direct_mult_1.5" 1.5 0.6

# availability_threshold sweep
echo "[$(date +%H:%M)] ## SWEEP 2: availability_threshold" >&2
restore_files
run_config "avail_thr_0.4" 1.2 0.4

restore_files
run_config "avail_thr_0.8" 1.2 0.8

# China policy_penalty sweep (modifies policy CSV)
echo "[$(date +%H:%M)] ## SWEEP 3: china_penalty" >&2
restore_files
set_china_penalty 150
run_config "penalty_150" 1.2 0.6

restore_files
set_china_penalty 600
run_config "penalty_600" 1.2 0.6

# delay_cost sweep (modifies scenario_transport_paths)
echo "[$(date +%H:%M)] ## SWEEP 4: delay_cost" >&2
restore_files
set_delay_cost 5.0
run_config "delay_5" 1.2 0.6

restore_files
set_delay_cost 10.0
run_config "delay_10" 1.2 0.6

# lithium_price_multiplier sweep (Fig 4 only)
echo "[$(date +%H:%M)] ## SWEEP 5: lithium_price_multiplier (Fig4 only)" >&2
restore_files
set_lithium_price_mult 3.0
run_config_fig4_only "li_price_3"

restore_files
set_lithium_price_mult 5.0
run_config_fig4_only "li_price_5"

# Final restore
restore_files
echo "[$(date +%H:%M)] ## ALL SENSITIVITY SWEEPS COMPLETE" >&2
