"""MEA CO2 CAPTURE — COMPLETE ML TRAINING PIPELINE.

This script loads the generated process dataset, trains ANN, Random Forest,
Gradient Boosting models, evaluates them, saves the best model, computes SHAP
feature importance, and performs a genetic algorithm optimisation.

It is intentionally compatible with both the project-style field names and the
training pipeline layout requested by the user.
"""

import os
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")

COLOURS = {
    "ANN": "#2196F3",
    "Random Forest": "#4CAF50",
    "Gradient Boosting": "#FF5722",
}
sns.set_style("whitegrid")


def prepare_dataset(project_dir: Path) -> pd.DataFrame:
    """Create a training dataset in the format expected by the ML pipeline."""
    source_csv = project_dir / "simulation_results.csv"
    dataset_csv = project_dir / "CO2_dataset.csv"

    if dataset_csv.exists():
        return pd.read_csv(dataset_csv)

    if not source_csv.exists():
        raise FileNotFoundError(
            f"Expected dataset at {source_csv}. Run the simulation script first."
        )

    df_src = pd.read_csv(source_csv)
    df = pd.DataFrame({
        "Flue_Gas_Flow_kmolhr": df_src["F_G"],
        "Inlet_CO2_mol%": df_src["y_CO2"],
        "Absorber_Temp_C": df_src["T_in"],
        "MEA_Concentration_wt%": df_src["C_MEA"],
        "LG_Ratio": df_src["L_G"],
        "CO2_Capture_Efficiency_%": df_src["capture_efficiency_pct"],
        "Reboiler_Duty_MJ_kgCO2": df_src["specific_reboiler_duty_MJ_kgCO2"],
    })

    df.to_csv(dataset_csv, index=False)
    return df


print("=" * 65)
print("   MEA CO₂ CAPTURE — ML TRAINING PIPELINE")
print("=" * 65)

project_dir = Path(__file__).resolve().parent
y_dir = project_dir / "y"
y_dir.mkdir(exist_ok=True)

# Prefer the dedicated results folder for final artifacts
models_dir = y_dir
models_dir.mkdir(exist_ok=True)

df = prepare_dataset(project_dir)
print(f"\n✓ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

FEATURES = [
    "Flue_Gas_Flow_kmolhr",
    "Inlet_CO2_mol%",
    "Absorber_Temp_C",
    "MEA_Concentration_wt%",
    "LG_Ratio",
]
TARGETS = ["CO2_Capture_Efficiency_%", "Reboiler_Duty_MJ_kgCO2"]
FEATURE_LABELS = [
    "Flue Gas Flow\n(kmol/hr)",
    "Inlet CO₂\n(mol%)",
    "Absorber Temp\n(°C)",
    "MEA Conc\n(wt%)",
    "L/G Ratio\n(kg/kg)",
]

x = df[FEATURES].values
y = df[TARGETS].values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
print("✓ Data split created (80/20 train/test)")

ann_grid = {
    "model__hidden_layer_sizes": [(64, 32), (128, 64, 32)],
    "model__activation": ["relu", "tanh"],
    "model__learning_rate_init": [1e-4, 1e-3],
    "model__alpha": [1e-5, 1e-4],
    "model__max_iter": [600, 1000],
}

rf_grid = {
    "model__estimator__n_estimators": [200, 300],
    "model__estimator__max_depth": [8, 12],
    "model__estimator__min_samples_split": [2, 4],
}

xgb_grid = {
    "model__estimator__n_estimators": [200, 300],
    "model__estimator__max_depth": [3, 6],
    "model__estimator__learning_rate": [0.03, 0.05],
    "model__estimator__subsample": [0.8],
}

models = {
    "ANN": (
        Pipeline([
            ("scaler", MinMaxScaler()),
            ("model", MLPRegressor(random_state=42, early_stopping=True, validation_fraction=0.1)),
        ]),
        ann_grid,
    ),
    "Random Forest": (
        Pipeline([
            ("scaler", MinMaxScaler()),
            ("model", MultiOutputRegressor(RandomForestRegressor(random_state=42, n_jobs=-1))),
        ]),
        rf_grid,
    ),
    "Gradient Boosting": (
        Pipeline([
            ("scaler", MinMaxScaler()),
            ("model", MultiOutputRegressor(xgb.XGBRegressor(random_state=42, verbosity=0))),
        ]),
        xgb_grid,
    ),
}


def evaluate(model, x_tr, x_te, y_tr, y_te):
    model.fit(x_tr, y_tr)
    y_pred = model.predict(x_te)
    metrics = {}
    for i, target in enumerate(TARGETS):
        yt = y_te[:, i]
        yp = y_pred[:, i]
        eps = 1e-9
        safe = np.abs(yt) > eps
        mape = np.mean(np.abs((yt[safe] - yp[safe]) / yt[safe])) * 100 if np.any(safe) else 0.0
        metrics[target] = {
            "R2": r2_score(yt, yp),
            "RMSE": np.sqrt(mean_squared_error(yt, yp)),
            "MAE": mean_absolute_error(yt, yp),
            "MAPE": mape,
        }
    return model, y_pred, metrics


print("\n── Training models with GridSearchCV ─────────────────────")
results = {}
preds = {}
best_r2 = -np.inf
best_name = ""
best_model = None

for name, (model, param_grid) in models.items():
    print(f"  Tuning {name}...", end=" ")
    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="r2",
        cv=2,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(x_train, y_train)
    best_estimator = search.best_estimator_
    y_pred = best_estimator.predict(x_test)
    met = {}
    for i, target in enumerate(TARGETS):
        yt = y_test[:, i]
        yp = y_pred[:, i]
        eps = 1e-9
        safe = np.abs(yt) > eps
        mape = np.mean(np.abs((yt[safe] - yp[safe]) / yt[safe])) * 100 if np.any(safe) else 0.0
        met[target] = {
            "R2": r2_score(yt, yp),
            "RMSE": np.sqrt(mean_squared_error(yt, yp)),
            "MAE": mean_absolute_error(yt, yp),
            "MAPE": mape,
        }
    results[name] = met
    preds[name] = y_pred
    avg_r2 = np.mean([met[t]["R2"] for t in TARGETS])
    print(f"best cv R² = {search.best_score_:.4f} | test avg R² = {avg_r2:.4f}")
    if avg_r2 > best_r2:
        best_r2 = avg_r2
        best_name = name
        best_model = best_estimator

joblib.dump(best_model, models_dir / "best_model.pkl")
joblib.dump({"name": best_name}, models_dir / "best_model_info.pkl")
print(f"\n✓ Best model: {best_name} (avg R² = {best_r2:.4f})")

rows = []
for name, met in results.items():
    for target, m in met.items():
        rows.append(
            {
                "Model": name,
                "Target": target,
                "R2": round(m["R2"], 4),
                "RMSE": round(m["RMSE"], 4),
                "MAE": round(m["MAE"], 4),
                "MAPE(%)": round(m["MAPE"], 2),
            }
        )
results_df = pd.DataFrame(rows)
results_df.to_csv(models_dir / "results_summary.csv", index=False)

print("\n── Model Performance Summary ───────────────────────────────")
print(results_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold", y=1.01)
metrics_to_plot = [("R2", True), ("RMSE", False)]

for ax, (metric, higher_better) in zip(axes, metrics_to_plot):
    for i, target in enumerate(TARGETS):
        vals = [results[n][target][metric] for n in models]
        x_pos = np.arange(len(models)) + i * 0.35
        bars = ax.bar(
            x_pos,
            vals,
            width=0.3,
            color=[COLOURS[n] for n in models],
            alpha=0.85,
            label=target if i == 0 else "",
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(np.arange(len(models)) + 0.175)
    ax.set_xticklabels(list(models.keys()), fontsize=10)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f"{metric} ({'Higher better' if higher_better else 'Lower better'})")
    ax.legend(TARGETS, fontsize=8)

from matplotlib.patches import Patch
legend_els = [Patch(facecolor=COLOURS[n], label=n) for n in models]
fig.legend(
    handles=legend_els,
    loc="lower center",
    ncol=3,
    bbox_to_anchor=(0.5, -0.08),
    fontsize=10,
)
plt.tight_layout()
plt.savefig(models_dir / "model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✓ Plot saved: model_comparison.png")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Actual vs Predicted — All Models", fontsize=14, fontweight="bold")

for col, (name, yp) in enumerate(preds.items()):
    for row, target in enumerate(TARGETS):
        ax = axes[row][col]
        y_t = y_test[:, row]
        y_p = yp[:, row]
        r2 = results[name][target]["R2"]
        mn, mx = min(y_t.min(), y_p.min()), max(y_t.max(), y_p.max())
        ax.scatter(y_t, y_p, alpha=0.6, color=COLOURS[name], s=25, edgecolors="none")
        ax.plot([mn, mx], [mn, mx], "k--", linewidth=1.2, label="Perfect fit")
        ax.set_xlabel(f"Actual {target}", fontsize=9)
        ax.set_ylabel("Predicted", fontsize=9)
        ax.set_title(f"{name}\n{target}  R²={r2:.4f}", fontsize=9)
        ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(models_dir / "parity_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Plot saved: parity_plots.png")

print(f"\n── SHAP Feature Importance ({best_name}) ───────────────────")
try:
    rf_model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        )
    )
    rf_model.fit(x_train, y_train)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("SHAP Feature Importance Analysis", fontsize=14, fontweight="bold")

    for idx, (target, ax) in enumerate(zip(TARGETS, axes)):
        est = rf_model.estimators_[idx]
        explainer = shap.TreeExplainer(est)
        shap_values = explainer(x_test)
        values = np.abs(shap_values.values).mean(axis=0)
        sorted_idx = np.argsort(values)

        ax.barh(
            [FEATURE_LABELS[i] for i in sorted_idx],
            values[sorted_idx],
            color=["#FF5722" if i == sorted_idx[-1] else "#2196F3" for i in range(len(FEATURE_LABELS))],
            edgecolor="white",
        )
        ax.set_xlabel("Mean |SHAP Value|", fontsize=10)
        ax.set_title(f"Feature Importance\n{target}", fontsize=10, fontweight="bold")
        ax.tick_params(axis="y", labelsize=9)

        print(f"\n  {target}:")
        for i in reversed(sorted_idx):
            print(f"    {FEATURE_LABELS[i].replace(chr(10), ' '):<30} SHAP = {values[i]:.4f}")

    plt.tight_layout()
    plt.savefig(models_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n✓ Plot saved: feature_importance.png")
except Exception as e:
    print(f"  SHAP skipped: {e}")

print("\n── Genetic Algorithm Optimisation ──────────────────────────")
print("  Objective: Maximise capture efficiency, Minimise reboiler duty")

BOUNDS = np.array([
    [500, 2000],
    [10, 20],
    [30, 60],
    [20, 40],
    [2, 6],
])

POP_SIZE = 100
N_GEN = 200
CX_PROB = 0.8
MUT_PROB = 0.1
W_ETA = 0.6
W_Q = 0.4


def fitness(individual):
    pred = best_model.predict([individual])[0]
    eta_n = (pred[0] - 30) / (99 - 30)
    q_n = (pred[1] - 2.5) / (5.5 - 2.5)
    return W_ETA * eta_n - W_Q * q_n


pop = BOUNDS[:, 0] + np.random.rand(POP_SIZE, 5) * (BOUNDS[:, 1] - BOUNDS[:, 0])

best_fitness_history = []
best_individual = None
best_fit = -np.inf

for gen in range(N_GEN):
    fit_scores = np.array([fitness(ind) for ind in pop])
    best_idx = np.argmax(fit_scores)

    if fit_scores[best_idx] > best_fit:
        best_fit = fit_scores[best_idx]
        best_individual = pop[best_idx].copy()
    best_fitness_history.append(best_fit)

    new_pop = []
    for _ in range(POP_SIZE):
        t_idx = np.random.choice(POP_SIZE, 3, replace=False)
        winner = t_idx[np.argmax(fit_scores[t_idx])]
        new_pop.append(pop[winner].copy())
    pop = np.array(new_pop)

    for i in range(0, POP_SIZE - 1, 2):
        if np.random.rand() < CX_PROB:
            pt = np.random.randint(1, 5)
            pop[i, pt:], pop[i + 1, pt:] = pop[i + 1, pt:].copy(), pop[i, pt:].copy()

    for i in range(POP_SIZE):
        if np.random.rand() < MUT_PROB:
            j = np.random.randint(5)
            pop[i, j] += np.random.normal(0, 0.05 * (BOUNDS[j, 1] - BOUNDS[j, 0]))
            pop[i, j] = np.clip(pop[i, j], BOUNDS[j, 0], BOUNDS[j, 1])

    if (gen + 1) % 50 == 0:
        print(f"  Generation {gen + 1:>3} — best fitness: {best_fit:.4f}")

opt_pred = best_model.predict([best_individual])[0]
print(f"\n  ── Optimisation Results ──────────────────────────────")
print(f"  Flue Gas Flow Rate   : {best_individual[0]:.1f} kmol/hr")
print(f"  Inlet CO₂ Conc       : {best_individual[1]:.2f} mol%")
print(f"  Absorber Temperature : {best_individual[2]:.2f} °C")
print(f"  MEA Concentration    : {best_individual[3]:.2f} wt%")
print(f"  L/G Ratio            : {best_individual[4]:.3f} kg/kg")
print(f"  ── Predicted Outputs ─────────────────────────────────")
print(f"  CO₂ Capture Efficiency : {opt_pred[0]:.2f} %")
print(f"  Specific Reboiler Duty : {opt_pred[1]:.4f} MJ/kg CO₂")

baseline = np.array([1000, 13, 40, 30, 4])
base_pred = best_model.predict([baseline])[0]
print(f"\n  ── Improvement vs Baseline ───────────────────────────")
print(f"  Capture efficiency: {base_pred[0]:.2f}% → {opt_pred[0]:.2f}%  "
      f"(+{opt_pred[0]-base_pred[0]:.2f}%)")
print(f"  Reboiler duty    : {base_pred[1]:.4f} → {opt_pred[1]:.4f} MJ/kg CO₂  "
      f"({((base_pred[1]-opt_pred[1])/base_pred[1])*100:.1f}% reduction)")

opt_df = pd.DataFrame(
    {
        "Parameter": [
            "Flue Gas Flow (kmol/hr)",
            "Inlet CO2 (mol%)",
            "Absorber Temp (°C)",
            "MEA Conc (wt%)",
            "L/G Ratio",
        ],
        "Baseline": [1000, 13, 40, 30, 4],
        "Optimised": np.round(best_individual, 3),
    }
)
opt_df.to_csv(models_dir / "optimisation_results.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(best_fitness_history, color="#2196F3", linewidth=2)
ax.fill_between(range(len(best_fitness_history)), best_fitness_history, alpha=0.15, color="#2196F3")
ax.set_xlabel("Generation", fontsize=11)
ax.set_ylabel("Best Fitness Score", fontsize=11)
ax.set_title("Genetic Algorithm Convergence", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(models_dir / "ga_convergence.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✓ Plot saved: ga_convergence.png")

print("\n" + "=" * 65)
print("   TRAINING COMPLETE — all models, plots and results saved")
print("=" * 65)
