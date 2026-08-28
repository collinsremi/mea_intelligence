import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb

sim = pd.read_csv('simulation_results.csv')
print('SIM COLUMNS:', sim.columns.tolist())
print(sim.head(3).to_string(index=False))

feature_sets = {
    'current_5': ['Flue_Gas_Flow_kmolhr','Inlet_CO2_mol%','Absorber_Temp_C','MEA_Concentration_wt%','LG_Ratio'],
    'full_9': ['F_G','y_CO2','T_in','L_G','L','C_MEA','T_reb','alpha_lean','P_abs'],
}

targets = {
    'current': ['CO2_Capture_Efficiency_%','Reboiler_Duty_MJ_kgCO2'],
    'full': ['capture_efficiency_pct','specific_reboiler_duty_MJ_kgCO2'],
}

for feature_name, feats in feature_sets.items():
    target_cols = targets['full'] if feature_name == 'full_9' else targets['current']
    X = sim[feats]
    y = sim[target_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {
        'xgb': MultiOutputRegressor(xgb.XGBRegressor(random_state=42, n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, objective='reg:squarederror', n_jobs=-1)),
        'rf': MultiOutputRegressor(RandomForestRegressor(random_state=42, n_estimators=500, n_jobs=-1)),
        'et': MultiOutputRegressor(ExtraTreesRegressor(random_state=42, n_estimators=800, n_jobs=-1)),
        'gbr': MultiOutputRegressor(GradientBoostingRegressor(random_state=42)),
        'hgb': MultiOutputRegressor(HistGradientBoostingRegressor(random_state=42)),
        'ann': Pipeline([('scaler', StandardScaler()), ('model', MLPRegressor(hidden_layer_sizes=(128,64), random_state=42, max_iter=4000, early_stopping=True))]),
    }
    print(f'\n=== {feature_name} ===')
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        print(f'--- {name} ---')
        for i, target in enumerate(y.columns):
            yt = y_test[target].to_numpy()
            yp = pred[:, i]
            print(target, 'R2=', round(r2_score(yt, yp), 4), 'RMSE=', round(np.sqrt(mean_squared_error(yt, yp)), 4), 'MAE=', round(mean_absolute_error(yt, yp), 4))
