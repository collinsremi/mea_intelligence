"""Custom amine-based post-combustion CO2 capture simulation.

This script implements a reduced-order, Python-based absorber/stripper model inspired by
 the methodology described in the project document:
  - 9 process variables sampled over a defined operating envelope
  - rate-based style mass transfer / kinetic effects represented through simplified equations
  - Latin Hypercube Sampling for input coverage
  - CSV export of the resulting process outputs

This is intentionally a custom engineering approximation, not a full commercial-scale
 electrolyte model. It is structured to match the project's workflow and output schema.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


VARIABLE_NAMES = [
    "F_G",
    "y_CO2",
    "T_in",
    "L_G",
    "L",
    "C_MEA",
    "T_reb",
    "alpha_lean",
    "P_abs",
]

BOUNDS = {
    "F_G": (80.0, 250.0),
    "y_CO2": (8.0, 16.0),
    "T_in": (30.0, 65.0),
    "L_G": (2.0, 6.5),
    "L": (160.6, 1450.4),
    "C_MEA": (20.0, 40.0),
    "T_reb": (105.0, 130.0),
    "alpha_lean": (0.15, 0.30),
    "P_abs": (0.90, 1.44),
}


def generate_latin_hypercube(n_samples: int, bounds: dict[str, tuple[float, float]], seed: int = 42) -> np.ndarray:
    """Generate a Latin Hypercube Sample for the nine design variables."""
    rng = np.random.default_rng(seed)
    variables = list(bounds.keys())
    samples = np.zeros((n_samples, len(variables)))

    for j, key in enumerate(variables):
        low, high = bounds[key]
        strata = np.arange(n_samples)
        random_offsets = rng.random(n_samples)
        # random permutation so each stratum is used once
        perm = rng.permutation(n_samples)
        loc = (perm + random_offsets) / n_samples
        samples[:, j] = low + (high - low) * loc

    return samples


def simulate_case(row: dict[str, float]) -> dict[str, float]:
    """Reduced-order process simulation for one case.

    The model captures the dominant process trends described in the methodology:
      - more solvent / higher L/G / higher MEA concentration improves absorption
      - higher inlet CO2 and higher gas flow push the load on the absorber
      - lower inlet temperature and lower lean loading improve capture
      - higher reboiler temperature increases regeneration energy demand
    """
    F_G = row["F_G"]
    y_CO2 = row["y_CO2"] / 100.0
    T_in = row["T_in"]
    L_G = row["L_G"]
    L = row["L"]
    C_MEA = row["C_MEA"]
    T_reb = row["T_reb"]
    alpha_lean = row["alpha_lean"]
    P_abs = row["P_abs"]

    # Dimensionless ratios relative to baseline design conditions.
    gas_load = F_G / 150.0
    co2_fraction = y_CO2 / 0.12
    solvent_strength = (L_G / 4.2) * (C_MEA / 30.0) * (L / 620.0)
    temperature_effect = 1.0 / (1.0 + 0.018 * max(T_in - 45.0, 0.0))
    pressure_effect = P_abs / 1.10

    # Reaction-equilibrium / kinetics-style term for the absorber.
    # Higher solvent strength and lower lean loading increase capture; higher temperature
    # decreases absorptive driving force.
    absorber_term = solvent_strength * co2_fraction * pressure_effect * temperature_effect
    lean_penalty = 1.0 / (1.0 + 2.5 * max(alpha_lean - 0.22, 0.0) / 0.08)
    capture_efficiency = 100.0 * (1.0 - np.exp(-1.65 * absorber_term / (1.0 + 0.9 * gas_load) * lean_penalty))
    capture_efficiency = float(np.clip(capture_efficiency, 45.0, 99.0))

    # Rich loading is tied to the absorber driving force and the lean loading specification.
    rich_loading = alpha_lean + 0.16 + 0.28 * (capture_efficiency / 100.0) * (1.0 + 0.25 * max(gas_load - 1.0, 0.0))
    rich_loading = float(np.clip(rich_loading, 0.18, 0.75))

    # Reboiler duty: a reduced-order estimate based on desorption heat, sensible heating,
    # and latent water evaporation. Higher reboiler temperature and higher lean loading increase duty.
    duty = (
        2.2
        + 0.8 * (T_reb - 105.0) / 25.0
        + 1.4 * max(alpha_lean - 0.15, 0.0) / 0.15
        + 1.1 * max(y_CO2 - 0.12, 0.0) / 0.04
        + 0.8 * max(1.0 - capture_efficiency / 100.0, 0.0) * 4.0
    )
    duty = float(np.clip(duty, 2.0, 7.0))

    return {
        "F_G": F_G,
        "y_CO2": row["y_CO2"],
        "T_in": T_in,
        "L_G": L_G,
        "L": L,
        "C_MEA": C_MEA,
        "T_reb": T_reb,
        "alpha_lean": alpha_lean,
        "P_abs": P_abs,
        "capture_efficiency_pct": capture_efficiency,
        "rich_CO2_loading_mol_mol": rich_loading,
        "specific_reboiler_duty_MJ_kgCO2": duty,
    }


def build_dataset(n_samples: int = 8000, seed: int = 42) -> pd.DataFrame:
    """Generate the final design-of-experiments dataset and return a pandas DataFrame."""
    samples = generate_latin_hypercube(n_samples, BOUNDS, seed=seed)
    records = []

    for i in range(n_samples):
        row = {name: float(samples[i, j]) for j, name in enumerate(VARIABLE_NAMES)}
        records.append(simulate_case(row))

    df = pd.DataFrame(records)
    return df


def main() -> None:
    output_path = Path(__file__).resolve().parent / "simulation_results.csv"
    df = build_dataset(n_samples=8000, seed=42)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} simulation runs at {output_path}")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
