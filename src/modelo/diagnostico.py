import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

RAIZ = Path(__file__).resolve().parents[2]
PROC = RAIZ / "datos" / "procesado"

df = pd.read_csv(PROC / "features.csv")
FEATS = [c for c in df.columns if c not in ("y", "usuario", "fecha")]

print("Promedio de cada variable: normales vs ataques")
print(df.groupby("y")[FEATS].mean().T.round(3))

print("\nAUC de cada variable sola (0.5 = no sirve)")
for f in FEATS:
    print(f"{f:20s} {roc_auc_score(df['y'], df[f]):.3f}")

novedad = [f for f in FEATS if f.endswith(("_nuevo", "_nueva"))]
s = df[novedad].sum(axis=1)
print("\nLínea base (suma de novedades): "
      f"AUC={roc_auc_score(df['y'], s):.3f} | AP={average_precision_score(df['y'], s):.3f} "
      f"| azar={df['y'].mean():.3f}")