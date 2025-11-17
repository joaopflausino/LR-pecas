import pandas as pd
import numpy as np
import re
import pickle
import json
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


ARQ = "Aluminio_522.xlsx"
FEATURES = ["peso_mp", "peso_peca", "cavaco", "comprimento"]
TARGET = "preco"


def to_float_br(x):
    """Converte número em formato Brasil para float."""
    if pd.isna(x):
        return np.nan
    s = re.sub(r"[^\d,.\-]", "", str(x))
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return np.nan

def mape_safe(y_true, y_pred):
    """MAPE % ignorando zeros."""
    y_true = np.array(y_true, float)
    y_pred = np.array(y_pred, float)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask])) * 100

def rmse(y_true, y_pred):
    y_true = np.array(y_true, float)
    y_pred = np.array(y_pred, float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

def calculate_aic(n, mse, k):
    """Calcula AIC (Akaike Information Criterion)."""
    return n * np.log(mse) + 2 * k

def calculate_r2_adj(r2, n, k):
    """Calcula R² ajustado."""
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)


print("\n" + "="*60)
print("TREINAMENTO DO MODELO RANDOM FOREST (PICKLE)")
print("="*60 + "\n")

print(f"Carregando dados de: {ARQ}")
df = pd.read_excel(ARQ)
df.columns = [c.strip().lower() for c in df.columns]

all_cols = FEATURES + [TARGET]
faltantes = [c for c in all_cols if c not in df.columns]
if faltantes:
    raise ValueError(f"Faltam colunas: {faltantes}")

for c in all_cols:
    df[c] = df[c].apply(to_float_br)

df = df.dropna(subset=all_cols).reset_index(drop=True)

df["faixa"] = np.where(df[TARGET] <= 5000, 0, 1)

print(f"Total de registros: {len(df)}")
print("\nQuantidade por faixa:")
print(df["faixa"].value_counts())


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
modelos_dict = {}

for fx in [0, 1]:
    subset = df[df["faixa"] == fx].reset_index(drop=True)

    X = subset[FEATURES].to_numpy(float)
    y = subset[TARGET].to_numpy(float)

    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)

    print(f"\n{'='*60}")
    print(f"Treinando Random Forest - Faixa {fx} ({'<=5000' if fx==0 else '>5000'})")
    print(f"{'='*60}")

    rf = RandomForestRegressor(
        n_estimators=600,
        random_state=42,
        n_jobs=-1,
        max_depth=None
    )
    rf.fit(X_imp, y)
    y_pred = rf.predict(X_imp)

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    rmse_val = rmse(y, y_pred)
    mape_val = mape_safe(y, y_pred)
    mse = np.mean((y - y_pred) ** 2)

    n = len(y)
    k = len(FEATURES)
    r2_adj = calculate_r2_adj(r2, n, k)
    aic = calculate_aic(n, mse, k)

    print(f"n = {len(y)}")
    print(f"R²          : {r2:.4f}")
    print(f"R² ajustado : {r2_adj:.4f}")
    print(f"AIC         : {aic:.2f}")
    print(f"MAE         : R$ {mae:,.2f}")
    print(f"RMSE        : R$ {rmse_val:,.2f}")
    print(f"MAPE        : {mape_val:.2f}%")

    modelos_dict[f'model_faixa_{fx}'] = rf
    modelos_dict[f'imputer_faixa_{fx}'] = imp
    modelos_dict['features'] = FEATURES

    if fx == 0:
        config = {
            'r_squared': float(r2),
            'r_squared_adj': float(r2_adj),
            'aic': float(aic),
            'n_observations': int(n),
            'target': TARGET,
            'predictors': FEATURES,
            'timestamp': timestamp
        }


model_filename = f"rf_model_faixa0_{timestamp}.pkl"

with open(model_filename, 'wb') as f:
    pickle.dump(modelos_dict, f)
print(f" Modelo salvo (PICKLE): {model_filename}")

config_filename = "model_config.json"
with open(config_filename, 'w') as f:
    json.dump(config, f, indent=2)
print(f"Configuração salva: {config_filename}")



print(f"   {model_filename}")
print(f"   {config_filename}")

