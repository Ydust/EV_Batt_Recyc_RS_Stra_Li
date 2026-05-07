# Main Figure Organization

## Core Statement

本研究量化收集、技术、经济选择与贸易政策可达性障碍，如何削减全球退役动力电池回收中真正可进入电池供应链的二次锂。

Recommended English framing:

```text
This study quantifies how collection, technology, economic choice, and trade-policy accessibility barriers reduce battery-supply-chain-available secondary lithium from global end-of-life EV battery recycling.
```

## Main Indicator

Use one unifying indicator throughout the main figure:

```text
battery-supply-chain-available secondary lithium equivalent
```

This is not simply physically recovered lithium. It is the amount of secondary lithium that remains available to the battery supply chain after collection, technical recovery, trade-policy accessibility, and economic-selection barriers.

## Figure Logic

The main figure should be organized as a barrier cascade:

```text
embedded lithium potential
  - capture / collection barrier
  - technology recovery barrier
  - trade-policy accessibility barrier
  - economic-selection barrier
= battery-supply-chain-available secondary lithium equivalent
```

This framing keeps trade policy central without claiming that all policy-affected lithium is physically destroyed. The policy mechanism is an accessibility loss: lithium may still be processed, but its access to preferred, low-friction, battery-supply-chain routes is reduced.

## Proposed Main Figure Panels

### Panel A. Barrier Cascade

Purpose: show the conceptual and quantitative cascade from embedded lithium potential to supply-chain-available secondary lithium.

Recommended visual:

```text
waterfall or Sankey-style cascade
```

Stages:

```text
Embedded Li potential
Capture / collection loss
Technology recovery loss
Trade-policy accessibility loss
Economic-selection loss
Battery-supply-chain-available secondary Li
```

Best year choices:

```text
2030, 2040, 2050
```

Data basis:

```text
trans/scenario_result/high_collection/baseline/barrier_decomposition/lithium_barrier_decomposition.csv
Figure_data/lithium_loss_comparison.csv
```

### Panel B. Loss Mechanisms Over Time

Purpose: compare the four mechanisms over milestone years.

Recommended visual:

```text
grouped bars or small-multiple bars
```

Categories:

```text
Capture / collection loss
Technology recovery loss
Trade-policy accessibility loss
Economic-selection loss
```

Important wording:

```text
Trade-policy accessibility loss should be reported as Li-equivalent accessibility loss, not as physical unrecovered lithium unless the model calculates a true reduction in recovered lithium.
```

### Panel C. Trade-Policy Accessibility Mechanism

Purpose: make the policy mechanism visually central.

Recommended visual:

```text
world route map
```

Show:

```text
open-access routes
policy-constrained routes
rerouted or downgraded lithium flows
top affected source-destination pairs
```

Use this panel to explain that policy constraints can reduce route quality, increase friction, and shift access even when final recovered lithium remains similar.

Data basis:

```text
Figure_data/joint_policy_technology/route_access_loss_by_route.csv
Figure_data/joint_policy_technology/route_access_loss_by_policy_year.csv
Figure_data/joint_policy_technology/lithium_loss_scenarios/lithium_loss_scenarios_routes.csv
```

### Panel D. Regional Exposure

Purpose: identify where the accessibility barrier matters most.

Recommended visual:

```text
ranked horizontal bars
```

Possible rankings:

```text
Top source countries by policy-induced accessibility loss
Top destination countries losing access to secondary lithium
Regions with largest share of affected recoverable lithium
```

Use a relative metric alongside kt Li-equivalent:

```text
affected Li-equivalent / embedded Li potential
affected Li-equivalent / collected Li
affected Li-equivalent / supply-chain available secondary Li
```

## Recommended Figure Title

```text
Barriers reducing battery-supply-chain-available secondary lithium
```

Chinese title:

```text
削减电池供应链可用二次锂的多重障碍
```

## Key Terminology

Use:

```text
trade-policy accessibility barrier
policy-induced accessibility loss
route-access disruption
battery-supply-chain-available secondary lithium equivalent
```

Avoid using:

```text
trade-policy physical loss
route displacement loss
lost lithium
```

unless the value is explicitly based on reduced recovered lithium.

## Interpretation Template

Suggested manuscript wording:

```text
The central outcome is not only how much lithium is physically unrecovered, but how much potentially recoverable secondary lithium remains accessible to the battery supply chain. Collection barriers reduce the material entering the recycling system; technology barriers reduce lithium recovered from processed material; trade-policy barriers reduce route accessibility and the quality of supply-chain access; and economic-selection barriers capture the gap between lithium-maximizing and realistic decision modes.
```

## Implementation Notes

Current scripts already cover parts of this figure:

```text
plot_figure_a_lithium_flow.py
plot_barrier_figures.py
plot_policy_transfer_route_maps.py
plot_route_access_loss_main.py
compare_four_lithium_losses.py
```

Recommended next coding step:

```text
Create plot_main_supply_chain_available_lithium.py
```

That script should merge:

```text
lithium_barrier_decomposition.csv
lithium_loss_comparison.csv
route_access_loss_by_route.csv
```

and write:

```text
Figure_data/main_supply_chain_available_lithium.csv
Figure_data/main_supply_chain_available_lithium.png
```
