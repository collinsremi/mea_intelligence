# Reduced-Order MEA CO2 Capture Methodology

This project does not implement a first-principles absorber/stripper model based on Kent-Eisenberg equilibrium, Onda mass-transfer correlations, Hikita/Austgen kinetics, or a differential packed-height solver. The model currently implemented in the repository is a reduced-order engineering surrogate designed for surrogate-model training and process optimization.

## Scope of the implemented model

The simulation builds a synthetic process dataset over a defined operating envelope using nine process variables:

- F_G: flue gas flow rate (kmol/hr)
- y_CO2: inlet CO2 concentration (mol%)
- T_in: absorber inlet temperature (°C)
- L_G: liquid-to-gas ratio
- L: solvent circulation rate (kg/hr)
- C_MEA: MEA concentration (wt%)
- T_reb: reboiler temperature (°C)
- alpha_lean: lean solvent loading (mol CO2/mol MEA)
- P_abs: absorber pressure (bar)

These variables are generated using Latin Hypercube Sampling across the operating ranges defined in the simulator.

## Actual implemented equations

The surrogate uses dimensionless process ratios to represent the dominant absorption and regeneration trends. The calculations in the code are as follows.

### 1. Process intensity terms

- `gas_load = F_G / 150.0`
- `co2_fraction = y_CO2 / 0.12`
- `solvent_strength = (L_G / 4.2) * (C_MEA / 30.0) * (L / 620.0)`
- `temperature_effect = 1 / (1 + 0.018 * max(T_in - 45, 0))`
- `pressure_effect = P_abs / 1.10`

### 2. Absorber response term

The absorber effect is represented by:

- `absorber_term = solvent_strength * co2_fraction * pressure_effect * temperature_effect`
- `lean_penalty = 1 / (1 + 2.5 * max(alpha_lean - 0.22, 0) / 0.08)`

The capture efficiency is then estimated by:

- `capture_efficiency = 100 * (1 - exp(-1.65 * absorber_term / (1 + 0.9 * gas_load) * lean_penalty))`

This result is clipped to a realistic operational range to keep output values within a usable envelope:

- `capture_efficiency = clip(capture_efficiency, 45, 99)`

### 3. Rich loading estimate

The rich loading is not computed from a mechanistic equilibrium model. It is estimated from the capture response and lean loading:

- `rich_loading = alpha_lean + 0.16 + 0.28 * (capture_efficiency / 100) * (1 + 0.25 * max(gas_load - 1, 0))`

with a clipped range of:

- `rich_loading = clip(rich_loading, 0.18, 0.75)`

### 4. Reboiler duty estimate

The specific reboiler duty is approximated from energy-related process trends:

- `duty = 2.2 + 0.8 * (T_reb - 105)/25 + 1.4 * max(alpha_lean - 0.15, 0)/0.15 + 1.1 * max(y_CO2 - 0.12, 0)/0.04 + 0.8 * max(1 - capture_efficiency/100, 0) * 4.0`

and clipped to a practical operating range:

- `duty = clip(duty, 2.0, 7.0)`

## Interpretation

This methodology is a data-driven engineering surrogate, not a mechanistic process simulator. It is appropriate for:

- generating a broad process-response dataset,
- benchmarking ML models,
- exploring operating trends and optimization candidates,
- rapid screening decisions within a project workflow.

It is not appropriate for:

- detailed absorber/stripper equipment design,
- rigorous equilibrium or mass-transfer calculations,
- legal or design-grade process specification,
- claiming the implementation of Kent-Eisenberg, Onda, or Hikita/Austgen models.

## Relation to the thesis/report wording

The methodology text in any formal document should clearly state that the project uses a reduced-order surrogate model informed by engineering trends and operational envelopes instead of a rigorous rate-based chemistry model. Any chapter claiming a full equilibrium-based packed-column model should be rewritten to reflect the actual code in this repository.
