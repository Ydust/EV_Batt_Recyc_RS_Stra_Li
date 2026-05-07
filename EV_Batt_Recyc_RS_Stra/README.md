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
```

## Data Sources

- **EV battery stock/scrap projections**: Dynamic material flow analysis
- **Trade data**: [UN Comtrade](https://comtrade.un.org/)
- **Distance data**: [CEPII GeoDist](https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=6)
- **Labor costs**: [ILO ILOSTAT](https://ilostat.ilo.org/)
- **Carbon tax rates**: World Bank Carbon Pricing Dashboard
- **Recycling cost/emission**: Literature-based life-cycle assessment data
