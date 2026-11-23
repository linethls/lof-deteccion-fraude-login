import numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import average_precision_score, roc_auc_score

RAIZ = Path(__file__).resolve().parents[2]
PROC = RAIZ / "datos" / "procesado"

df = pd.read_csv(PROC / "features.csv")
FEATS = [c for c in df.columns if c not in ("y", "usuario", "fecha")]
X_tr = np.load(PROC / "X_train_s.npy"); y_tr = np.load(PROC / "y_train.npy")
X_va = np.load(PROC / "X_val_s.npy");   y_va = np.load(PROC / "y_val.npy")

# Línea base en validación (suma de novedades, columnas escaladas)
nov = [i for i, f in enumerate(FEATS) if f.endswith(("_nuevo", "_nueva"))]
base = X_va[:, nov].sum(axis=1)
print(f"Línea base: AP={average_precision_score(y_va, base):.3f} "
      f"AUC={roc_auc_score(y_va, base):.3f} | azar={y_va.mean():.3f}\n")

def probar(nombre, cols, pesos=None):
    A, B = X_tr[:, cols], X_va[:, cols]
    if pesos is not None:
        A, B = A * pesos, B * pesos
    A = A[y_tr == 0]
    for k in [20, 50, 100]:
        lof = LocalOutlierFactor(n_neighbors=k, novelty=True, n_jobs=-1).fit(A)
        s = -lof.score_samples(B)
        print(f"{nombre:28s} k={k:3d} AP={average_precision_score(y_va, s):.3f} "
              f"AUC={roc_auc_score(y_va, s):.3f}")

todas = list(range(len(FEATS)))
sin_cont = [i for i, f in enumerate(FEATS) if f not in ("seg_desde_ultimo", "login_ok")]
pesos = np.array([3.0 if FEATS[i].endswith(("_nuevo", "_nueva")) else 1.0
                  for i in sin_cont])

probar("todas las variables", todas)
probar("sin seg_ni login_ok", sin_cont)
probar("sin seg_ni login_ok, pesos x3", sin_cont, pesos)