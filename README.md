# Global EV Battery Recycling Strategy Optimization

## Overview

This project evaluates **global recycling strategies for end-of-life electric vehicle (EV) batteries (excluding China)**. It constructs a cost–benefit optimization model to compare different recycling pathways and cross-border transportation strategies, assessing net profits, recycling costs, and CO₂ emissions for countries worldwide from **2023 to 2033**.

## Research Framework

### Battery Recycling Methods

Three recycling technologies are evaluated:

- **Pyrometallurgy (Pyro)** — smelting-based recovery
- **Hydrometallurgy (Hydro)** — chemical leaching-based recovery
- **Direct Recycling (Direct)** — cathode material regeneration

An **Optimal** mode selects the best recycling method per country–battery combination to maximize net profits.

### Recycling Strategies

| Strategy | Description |
|----------|-------------|
| **Strategy 1** | Each country recycles its own retired EVBs domestically |
| **Strategy 2** | Cross-border recycling among EV-producing countries with existing recycling capacity |
| **Strategy 3** | Global recycling network including non-producer countries, using trade-probability-based and linear-programming-optimized transportation |

### Battery Types

Six lithium-ion battery chemistries are modeled: **NMC811, NMC111, NMC523, NMC622, NCA, LFP**.

## Project Structure

### Core Scripts

| File | Description |
|------|-------------|
| `fun_class.py` | Utility functions: trade probability computation, Monte Carlo transport simulation, distance matrix construction, **linear programming (LP)** optimization (`scipy.optimize.linprog`), net profit calculation for each strategy, and power-function cost fitting |
| `cul_path_trans.py` | Computes recycling costs per country (labor, utilities, materials, carbon tax, general expenses), fits cost curves, and calculates optimal transportation plans for Strategy 2 & 3 using specific recycling methods (Pyro/Hydro/Direct) |
| `cul_path_trans_optimal.py` | Same pipeline as `cul_path_trans.py` but uses the **Optimal** recycling mode (selects the best method per country–battery pair) |
| `output_net_profits.py` | Aggregates net profits, total costs, and CO₂ emissions across all strategies and years; generates grouped waterfall charts |
| `output_strategy_path.py` | Computes and exports per-country net profit results for the optimal recycling strategy selection across all years |
| `picture_out.py` | World map visualizations showing recycling method selection and net profit distribution by country |
| `picture_output.ipynb` | Jupyter notebook with comprehensive figure generation (bar charts, world maps, stacked areas, Sankey-style flows, etc.) |
| `Batch_export.py` | Batch sensitivity analysis: runs the full pipeline across different scrap mass scenarios (parameter sweep on Weibull shape parameter 2.2–3.0) |
| `recycling_rate_scenarios.py` | Country-level recycling-rate scenario module: estimates collected scrap, recycled lithium, uncollected lithium, and primary lithium offset before running the full network optimization |
| `scenario_transport_paths.py` | Re-solves scenario-specific Strategy 2/3 transportation paths and writes them to `trans/scenario_paths/` |
| `technology_choice_modes.py` | Post-processes Strategy 1/2/3 outputs to compare technology choices under profit-maximizing, lithium-maximizing, and multi-objective selection modes |

### Cost Model Components (`cost/`)

| File | Content |
|------|---------|
| `cost_breakdown.csv` | Per-country recycling cost breakdown (utility, materials, general expenses) |
| `Combined_labor_cost.csv` | ILO hourly labor cost data by country |
| `equ_cost.csv` | Equipment person-hours per recycling method |
| `carbon_tax_nation.csv` | Carbon tax rates by jurisdiction (US$/tCO₂e) |
| `recycling_emission_count.csv` | CO₂ emissions per battery type and recycling method (g/kg) |
| `materials value.csv` | Market value of recovered materials ($/kg) |
| `Metal content.csv` | Metal content fractions per battery type |
| `Produced materials for recycling.csv` | Material output per recycling method and battery type |
| `Recovery efficiency.csv` | Recovery efficiency by recycling method |
| `default_cot.csv` | Baseline (UK) recycling cost curve |
| `cost_coun_df.csv` | Computed country-level cost data |

### Scenario Data (`Scenario result/`)

| File | Content |
|------|---------|
| `EV_battery_inuse_scrap.csv` | Retired EV battery scrap mass by country, year, and battery type |
| `EV_battery_inuse_scrap_total.csv` | Total retired EV battery scrap mass by country and year |
| `EV_battery_stock.csv` | In-use EV battery stock by type |

### Recycling-Rate Scenario Extension

The recycling-rate extension is designed as a two-step add-on.

**Step 1: stand-alone lithium impact analysis**

Run:

```bash
python recycling_rate_scenarios.py
```

If `country_recycling_rate_scenarios.csv` does not exist, the script creates a default country-year table with three scenarios:

- `baseline`
- `high_collection`
- `low_collection`

The rate table can be edited directly. Required columns:

```text
Year,country,scenario,collection_rate
```

where `collection_rate` is between 0 and 1 and `country` must match the `region` names in `Scenario result/EV_battery_inuse_scrap.csv`.

Outputs are written to `Scenario result/recycling_rate/`:

| File | Content |
|------|---------|
| `lithium_recycling_rate_detail.csv` | Country-year-scenario-battery-method results for retired lithium, collected lithium, recycled lithium, uncollected lithium, and primary lithium offset |
| `lithium_recycling_rate_country_summary.csv` | Country-level summary by year, scenario, and recycling method |
| `lithium_recycling_rate_global_summary.csv` | Global summary by year, scenario, and recycling method |
| `EV_battery_inuse_scrap_collected_by_scenario.csv` | Scenario-adjusted scrap input for linking into the full transport and cost-benefit model |

Lithium recovery efficiencies for technology routes are stored separately in:

```text
technology_lithium_recovery_scenarios.csv
```

Required columns:

```text
recycling_m,recovery_efficiency_scenario,li_recovery_efficiency,source_basis,notes
```

The default values are literature/public-data scenarios:

| Route | Low | Baseline | High | Rationale |
|-------|-----|----------|------|-----------|
| Pyro | 0.00 | 0.10 | 0.30 | Conventional pyrometallurgy mainly recovers Ni/Co/Cu; Li often reports to slag and requires additional treatment |
| Hydro | 0.80 | 0.90 | 0.95 | Hydrometallurgical routes are commonly characterized as high Li-recovery routes; 0.80 is aligned with reported overall Li recovery cases and EU 2031 Li recovery target |
| Direct | 0.80 | 0.90 | 0.95 | Direct recycling preserves/regenerates cathode active material, but lower cases reflect feedstock sorting, degradation, and scale-up uncertainty |

The scenario module outputs both the country collection-rate scenario (`scenario`) and the technology recovery-efficiency scenario (`recovery_efficiency_scenario`).

Dynamic driver tables can be used instead of manually entering every year:

```text
country_recycling_rate_drivers.csv
technology_recovery_efficiency_drivers.csv
```

If these driver tables exist, `recycling_rate_scenarios.py` uses logistic growth curves to generate the annual parameter tables automatically, then writes the generated values back to:

```text
country_recycling_rate_scenarios.csv
technology_lithium_recovery_scenarios.csv
```

Country collection-rate drivers use:

```text
country_group,scenario,start_rate,max_rate,midpoint,k,policy_year,policy_boost
```

Technology-efficiency drivers use:

```text
recycling_m,recovery_efficiency_scenario,start_eff,max_eff,midpoint,k
```

The dynamic collection-rate model is:

```text
collection_rate = start_rate + (max_rate - start_rate) / (1 + exp(-k * (Year - midpoint))) + policy_boost_after_policy_year
```

The dynamic technology-efficiency model is:

```text
li_recovery_efficiency = start_eff + (max_eff - start_eff) / (1 + exp(-k * (Year - midpoint)))
```

**Step 2: network optimization linkage**

Use `EV_battery_inuse_scrap_collected_by_scenario.csv` as the supply-side input to the existing strategy scripts. The `scrap` column is already adjusted by `collection_rate`, while `scrap_original` and `scrap_uncollected` are retained for reporting.

The existing strategy scripts keep their default behavior unless `RECYCLING_RATE_SCENARIO` is set:

```bash
# Windows PowerShell example
$env:RECYCLING_RATE_SCENARIO="baseline"
$env:RECOVERY_EFFICIENCY_SCENARIO="baseline"
$env:SCENARIO_YEAR_START="2025"
$env:SCENARIO_YEAR_END="2030"
# Or run selected non-contiguous years:
$env:SCENARIO_YEARS="2025,2030,2035,2040,2045,2050"
python cul_path_trans.py
python cul_path_trans_optimal.py
```

Valid collection scenario names are the values in `country_recycling_rate_scenarios.csv`, for example `baseline`, `high_collection`, and `low_collection`. Valid recovery-efficiency scenario names are the values in `technology_lithium_recovery_scenarios.csv`, for example `baseline`, `high`, and `low`.

When either scenario variable is set, the strategy scripts write results to scenario-specific folders:

```text
trans/scenario_result/{collection_scenario}/{recovery_efficiency_scenario}/
```

They also look first for scenario-specific transportation plans in:

```text
trans/scenario_paths/{collection_scenario}/{recovery_efficiency_scenario}/
```

If scenario-specific paths are not present, the scripts fall back to the existing `trans/paths/` transportation plans. This is the first-stage trade linkage: collection-rate-adjusted scrap and lithium metrics enter the Strategy 1/2/3 reporting layer without overwriting the original results. Re-solving the transportation linear programs into `trans/scenario_paths/` is the next integration step.

Scenario-enabled strategy outputs add:

```text
collection_scenario
recovery_efficiency_scenario
contained_lithium
li_recovery_efficiency
recycled_lithium
primary_lithium_offset
primary_lithium_gap
```

Technology-choice modes can then be compared with:

```bash
python technology_choice_modes.py --collection-scenario high_collection --recovery-scenario baseline --year-start 2025 --year-end 2025

# Or process selected long-term milestone years:
python technology_choice_modes.py --collection-scenario high_collection --recovery-scenario baseline --years 2025,2030,2035,2040,2045,2050
```

### Manufacturing Scrap Mass Submodel

The manufacturing-scrap extension estimates production-process scrap mass before the end-of-life battery flow enters the recycling network.

Run:

```bash
python generate_manufacturing_scrap_parameters.py
python generate_manufacturing_scrap_mass.py
```

Inputs:

```text
production_cap.csv
manufacturing_scrap_parameters.csv
2024-03-20 - Global Lithium-Ion Battery Supply Chain Ranking Dataset.xlsm
```

By default, `generate_manufacturing_scrap_mass.py` estimates actual manufacturing output from consumption/new battery placement rather than from nameplate capacity. The global activity level is inferred from the stock balance:

```text
consumption_proxy_t = stock_t - previous_stock_t + eol_scrap_t
```

That global consumption proxy is then allocated across manufacturing countries using annual production-capacity shares from `production_cap.csv`; 2022/2023 shares use BloombergNEF cell-capacity values from the workbook's `BatteryManufacturing` sheet when available. If a country's capacity series ends before 2050, the script carries forward the last positive capacity observation to 2050 and marks those rows in `capacity_source` as `forecast_hold_last_from_<year>`. Use `--activity-basis capacity` to reproduce the older capacity-based activity assumption, or `--capacity-source project` to run shares only with `production_cap.csv`.

The mass conversion uses a default finished-battery mass intensity of `0.006 t/kWh`:

```text
finished_battery_output_t = consumption_proxy_t * production_share
gross_manufacturing_scrap_t = finished_battery_output_t * manufacturing_scrap_rate / battery_manufacturing_yield
captured_manufacturing_scrap_t = gross_manufacturing_scrap_t * manufacturing_scrap_capture_rate
accepted_recycled_material_t = captured_manufacturing_scrap_t * recycled_material_acceptance_yield
```

Outputs:

| File | Content |
|------|---------|
| `manufacturing_scrap_mass.csv` | Country-year manufacturing scrap mass, capture, uncaptured scrap, accepted recycled-material mass, and capacity source |
| `manufacturing_scrap_mass_summary.csv` | Global year-level summary of production capacity and manufacturing scrap mass |
| `Scenario result/EV_battery_manufacturing_scrap.csv` | Main-model-compatible production scrap supply by year, producing country, and battery type |
| `Scenario result/EV_battery_manufacturing_scrap_total.csv` | Country-year totals for manufacturing scrap |
| `Scenario result/EV_battery_inuse_and_manufacturing_scrap.csv` | Existing EOL scrap plus manufacturing scrap in the main model row format |
| `Scenario result/EV_battery_inuse_and_manufacturing_scrap_total.csv` | Country-year totals for the combined EOL plus manufacturing scrap supply |

The main-model-compatible manufacturing output uses the same core columns as `Scenario result/EV_battery_inuse_scrap.csv`:

```text
Year,region,type,scrap,inuse,f_ebp,tau_w,f_tau
```

Manufacturing scrap is assigned only to countries with positive battery production capacity in `production_cap.csv`. Global BNEF `Production scrap` is allocated in two stages by default: first to BNEF regions using the `Figure 3` regional recycling-availability shares (`China`, `Europe`, `Japan`, `South Korea`, `US`, `ROW`), then to producing countries within each region by annual production-capacity share. The allocated country totals are then split by battery type using the project stock-balance chemistry shares. Non-producing countries do not receive manufacturing scrap. Use `--production-scrap-allocation global-capacity` to reproduce the simpler direct global-capacity allocation.

Sensitivity runs can override battery mass intensity:

```bash
python generate_manufacturing_scrap_mass.py --battery-mass-t-per-kwh 0.0055
```

Recommended full scenario workflow:

```bash
# 1. Generate dynamic collection-rate and technology-efficiency scenario tables.
python recycling_rate_scenarios.py

# 2. Re-solve scenario-specific Strategy 2/3 transport paths.
python scenario_transport_paths.py --collection-scenario high_collection --recovery-scenario baseline --years 2025,2030,2035,2040,2045,2050

# 3. Run strategy reporting against scenario paths and scenario scrap.
$env:RECYCLING_RATE_SCENARIO="high_collection"
$env:RECOVERY_EFFICIENCY_SCENARIO="baseline"
$env:SCENARIO_YEARS="2025,2030,2035,2040,2045,2050"
python cul_path_trans_optimal.py

# 4. Apply profit, lithium, multi-objective, and realistic technology-choice modes.
python technology_choice_modes.py --collection-scenario high_collection --recovery-scenario baseline --years 2025,2030,2035,2040,2045,2050

# 5. Decompose lithium entry barriers and regenerate the barrier-centered figure data.
python barrier_decomposition.py --collection-scenario high_collection --recovery-scenario baseline --years 2025,2030,2035,2040,2045,2050
python plot_barrier_figures.py --collection-scenario high_collection --recovery-scenario baseline --years 2025,2030,2035,2040,2045,2050 --selected-mode Realistic_multiobjective --strategy "Strategy 1"
```

Outputs are written to:

```text
trans/scenario_result/{collection_scenario}/{recovery_efficiency_scenario}/technology_choice_modes/
```

Choice modes:

```text
Optimal_profit         # original net-profit-maximizing choice
Optimal_lithium        # maximum recycled lithium
Optimal_multiobjective # normalized profit + alpha * lithium - beta * recycling CO2
Realistic_multiobjective # multi-objective score with country capability, technology maturity, availability, complexity, policy, and battery-type fit constraints
```

The realistic choice mode uses two additional editable parameter tables:

```text
technology_country_capability.csv
technology_battery_fit.csv
```

`technology_country_capability.csv` controls route availability and country-group readiness by year:

```text
country_group,recycling_m,year,availability,maturity_score,capability_score,complexity_penalty,policy_bonus
```

`technology_battery_fit.csv` controls battery chemistry fit:

```text
type,recycling_m,battery_type_fit
```

For `Realistic_multiobjective`, technologies with `availability` below `--availability-threshold` are excluded for that country group, year, and route. The default threshold is `0.5`.

### Lithium Entry Barrier Decomposition

The paper's main indicator is now:

```text
battery-supply-chain-available secondary lithium equivalent
```

This is a lithium-mass-equivalent supply-chain availability indicator. It does not imply that all material has been refined into battery-grade lithium compounds.

Run:

```bash
python barrier_decomposition.py --collection-scenario high_collection --recovery-scenario baseline --years 2025,2030,2035,2040,2045,2050
```

Outputs are written to:

```text
trans/scenario_result/{collection_scenario}/{recovery_efficiency_scenario}/barrier_decomposition/
```

Key columns:

```text
embedded_li
li_collected
collection_loss
capacity_mismatch_loss
technology_loss
trade_policy_loss
economic_selection_loss
supply_chain_available_secondary_li
raw_supply_chain_available_secondary_li
unit_consistency_flag
```

`economic_selection_loss` is calculated as the lithium-maximizing recycled lithium minus the recycled lithium under each selected mode, bounded by the physically available collected lithium. Raw technology-choice values are retained in `raw_*` columns so unit inconsistencies in upstream outputs remain visible rather than silently hidden.

Generate the five barrier-centered figure datasets and PNGs with:

```bash
python plot_barrier_figures.py --collection-scenario high_collection --recovery-scenario baseline --years 2025,2030,2035,2040,2045,2050 --selected-mode Realistic_multiobjective --strategy "Strategy 1"
```

Figure outputs are written to:

```text
Figure_data/barrier_decomposition/
```

The dynamic recycled-vs-primary lithium figure can be regenerated with configurable comparison dimensions:

```bash
# Recommended for the revised barrier storyline:
# rows = collection scenarios, columns = technologies, fixed recovery path = baseline.
python plot_dynamic_recycling_rate_comparison.py --row-dimension collection --fixed-recovery-scenario baseline --primary-gap-basis embedded --years 2025,2030,2035,2040,2045,2050

# Original-style comparison:
# rows = technology Li recovery paths, columns = technologies, fixed collection scenario = baseline.
python plot_dynamic_recycling_rate_comparison.py --row-dimension recovery --fixed-collection-scenario baseline --primary-gap-basis collected --years 2025,2030,2035,2040,2045,2050
```

The recommended `embedded` gap basis treats unrecovered primary demand as the gap against total embedded lithium potential, so the grey area includes collection losses as well as technology losses. The `collected` gap basis is narrower and compares only against collected lithium potential.

Policy sensitivity should be reported as scenario comparisons rather than a single legal claim about China exports:

```bash
python scenario_transport_paths.py --collection-scenario high_collection --recovery-scenario baseline --policy-scenario reference_policy --years 2025,2030,2035,2040,2045,2050
python scenario_transport_paths.py --collection-scenario high_collection --recovery-scenario baseline --policy-scenario current_policy --years 2025,2030,2035,2040,2045,2050
python scenario_transport_paths.py --collection-scenario high_collection --recovery-scenario baseline --policy-scenario strict_policy --years 2025,2030,2035,2040,2045,2050
```

`waste_trade_policy_constraints.csv` includes `source_basis` to distinguish regulation proxies from scenario assumptions. The strict China rule is a conservative sensitivity case, not a baseline statement that exports are absolutely forbidden.

Policy scenario names are standardized as:

```text
reference_policy      # low-friction reference policy, previously open_policy
current_policy        # current/common policy setting, previously baseline
strict_policy         # general strict policy sensitivity
critical_route_policy # stress test that restricts actually used high-flow routes
```

Older names (`open_policy`, `baseline`, `critical_route_strict_policy`) are retained only for historical result compatibility. New runs should use the standardized names above so that the word `baseline` is reserved for collection/recovery/model baseline assumptions.

### EV and Stationary Storage Policy Alignment

Country policies are tracked separately for two model objects:

```text
ev_power_battery
stationary_storage_battery
```

The source policy rules are stored in:

```text
battery_policy_alignment.csv
```

This table distinguishes regulation, official targets, policy strategies, stewardship schemes, and scenario assumptions through `policy_status` and `source_basis`. It includes policy signals such as EPR strength, traceability, battery passport requirements, lithium recovery floors, recycled lithium content requirements, storage deployment targets, EV sales targets, domestic-content pressure, and hazardous-transport penalties.

Expand the source rules into model-ready country-year parameters with:

```bash
python generate_policy_alignment.py --years 2025,2027,2030,2035,2040,2045,2050
```

Outputs are written to:

```text
Scenario result/policy_alignment/battery_policy_alignment_expanded.csv
Scenario result/policy_alignment/battery_policy_country_year_summary.csv
```

The summary table provides the parameters that can be linked to barrier mechanisms:

```text
collection_loss                  <- collection_rate_floor, epr_strength, traceability_requirement
technology_loss                  <- li_recovery_efficiency_floor
trade_policy_loss                <- hazardous_transport_penalty
economic_selection_loss          <- recycled_content_li_min, domestic_content_pressure, battery_passport_requirement
capacity_mismatch_loss           <- storage_deployment_target_gwh, ev_sales_target
```

EU battery rules are applied to current EU member countries in the country list. China EV-battery rules are treated as strong traceability/recovery policy for power batteries, while China storage policy is treated as a storage-demand expansion target rather than a current hard recycling mandate. Japan's 2035 electrified-vehicle target includes HEVs, so it should be interpreted as weaker than a pure BEV mandate.
| `EV_battery_stock_total.csv` | Total in-use EV battery stock |

### Trade & Distance Data

| File | Content |
|------|---------|
| `trade_in_con/` | UN Comtrade import/export data (HS codes: 850760, 850790, 870360, 870380) for trade flow probability estimation |
| `dist_nation.csv` | Bilateral distance matrix between countries (CEPII GeoDist) |
| `nation_list_new.csv` | Country metadata (ISO3, region, sub-region, continent) |
| `all_countries.csv` | Country clustering results (K-Means) with coordinates |

### Transportation Results (`trans/`)

- `trans/paths/` — Optimal transportation plans per year, strategy, and recycling method (LP solutions)
- `trans/result/` — Aggregated results: net profits, costs, CO₂ emissions per country for each year (2023–2033)

### Figure Data (`Figure_data/`)

Pre-computed CSV files for reproducing publication figures (Figure 1–5).

## Key Methods

1. **Cost Estimation**: Country-specific recycling costs are derived from a UK baseline adjusted by regional labor costs, carbon taxes, and material/utility costs. Costs are fitted to power functions of recycling capacity: $C(x) = a \cdot x^b$

2. **Transportation Optimization**: Cross-border battery scrap allocation is solved via **linear programming** minimizing total cost (transportation + recycling + carbon cost), subject to supply–demand balance constraints.

3. **Trade Flow Estimation**: Historical UN Comtrade data is used to estimate trade probabilities between EV-producing countries, projected forward via curve fitting.

4. **Net Profit Calculation**: $\text{Net Profit} = \text{Revenue from recovered materials} - \text{Recycling cost} - \text{Carbon cost} - \text{Transportation cost}$

5. **CO₂ Emissions**: Transport emissions computed at 0.017 kg CO₂/ton-mile; recycling process emissions from life-cycle inventory data.

## Dependencies

- Python 3.x
- `pandas`, `numpy`, `scipy` (optimization & curve fitting)
- `geopandas`, `matplotlib`, `mpl_toolkits.basemap`
- `brewer2mpl` (color palettes)

## Usage

```bash
# 1. Compute recycling costs and optimal transportation plans
python cul_path_trans.py           # For specific recycling methods
python cul_path_trans_optimal.py   # For optimal method selection

# 2. Aggregate net profits and emissions
python output_net_profits.py
python output_strategy_path.py

# 3. Generate visualizations
python picture_out.py
# Or use the Jupyter notebook for interactive figures:
jupyter notebook picture_output.ipynb

# 4. Sensitivity analysis (batch)
python Batch_export.py

# 5. Recycling-rate scenario analysis
python recycling_rate_scenarios.py
```

## Data Sources

- **EV battery stock/scrap projections**: Dynamic material flow analysis
- **Trade data**: [UN Comtrade](https://comtrade.un.org/)
- **Distance data**: [CEPII GeoDist](https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=6)
- **Labor costs**: [ILO ILOSTAT](https://ilostat.ilo.org/)
- **Carbon tax rates**: World Bank Carbon Pricing Dashboard
- **Recycling cost/emission**: Literature-based life-cycle assessment data
