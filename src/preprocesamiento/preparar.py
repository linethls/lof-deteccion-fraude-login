import numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[2]
PROC = RAIZ / "datos" / "procesado"
MODELOS = RAIZ / "modelos"
MODELOS.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PROC / "features.csv")
FEATS = [c for c in df.columns if c not in ("y", "usuario")]
X, y, g = df[FEATS].values, df["y"].values, df["usuario"].values

assert np.isfinite(X).all(), "Hay NaN o inf en las features"
print("Tomas de cuenta en total:", y.sum(), "de", len(y))

def partir(idx, n_splits):
    sg = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    a, b = next(sg.split(X[idx], y[idx], g[idx]))
    return idx[a], idx[b]

idx = np.arange(len(df))
i_trval, i_te = partir(idx, 5)      # ~20 % test
i_tr, i_va = partir(i_trval, 4)     # ~20 % validación

# Comprobación: ningún usuario en dos bloques
assert not (set(g[i_tr]) & set(g[i_te]))
assert not (set(g[i_tr]) & set(g[i_va]))
assert not (set(g[i_va]) & set(g[i_te]))

scaler = StandardScaler().fit(X[i_tr])   # solo con train
for nombre, ix in [("train", i_tr), ("val", i_va), ("test", i_te)]:
    np.save(PROC / f"X_{nombre}_s.npy", scaler.transform(X[ix]))
    np.save(PROC / f"y_{nombre}.npy", y[ix])
    print(f"{nombre}: {len(ix)} filas | ataques: {y[ix].sum()}")

joblib.dump(scaler, MODELOS / "scaler.joblib")