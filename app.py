import os
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "y" / "best_model.pkl"

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"), static_folder=str(BASE_DIR / "static"))

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run train_models.py first.")

model = joblib.load(MODEL_PATH)

FEATURES = [
    "Flue_Gas_Flow_kmolhr",
    "Inlet_CO2_mol%",
    "Absorber_Temp_C",
    "MEA_Concentration_wt%",
    "LG_Ratio",
]


def predict_case(payload: dict) -> dict:
    values = []
    for feature in FEATURES:
        values.append(float(payload.get(feature, 0.0)))

    x = np.array([values], dtype=float)
    pred = model.predict(x)[0]
    capture = float(pred[0])
    duty = float(pred[1])

    flow = float(payload.get("Flue_Gas_Flow_kmolhr", 1000.0))
    co2_in = float(payload.get("Inlet_CO2_mol%", 13.0))
    co2_in_kmol_hr = flow * (co2_in / 100.0)
    co2_removed_kmol_hr = co2_in_kmol_hr * (capture / 100.0)
    co2_removed_kg_hr = co2_removed_kmol_hr * 44.01

    return {
        "capture_efficiency_pct": round(capture, 2),
        "reboiler_duty_mj_kg_co2": round(duty, 4),
        "estimated_co2_removed_kg_hr": round(co2_removed_kg_hr, 2),
        "estimated_co2_removed_kmol_hr": round(co2_removed_kmol_hr, 4),
        "reboiler_total_mj_hr": round(duty * co2_removed_kg_hr, 2),
        "inputs": payload,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True)
    return jsonify(predict_case(payload))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
