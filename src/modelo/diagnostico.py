import sys
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
from formato import titulo, seccion, nota, tabla, miles

PROC = RAIZ / "datos" / "procesado"



df = pd.read_csv(PROC / "features.csv")
FEATS = [c for c in df.columns if c not in ("y", "usuario", "fecha")]
y = df["y"]
medias = df.groupby("y")[FEATS].mean()


def lectura(auc):
    d = abs(auc - 0.5)
    if d >= 0.20: return "buena"
    if d >= 0.10: return "regular"
    if d >= 0.05: return "débil"
    return "sin señal"


filas = []
for f in FEATS:
    auc = roc_auc_score(y, df[f])
    filas.append({"Variable": f, "Prom. normales": medias.loc[0, f],
                  "Prom. ataques": medias.loc[1, f],
                  "AUC sola": auc, "Señal": lectura(auc)})
filas.sort(key=lambda r: -abs(r["AUC sola"] - 0.5))

seccion("Cada variable por separado (ordenadas de mayor a menor señal)")
tabla(pd.DataFrame(filas), izquierda=("Variable", "Señal"))
print("Criterio de la columna Señal: |AUC - 0.5| >= 0.20 buena, >= 0.10 regular, "
      ">= 0.05 débil.")

novedad = [f for f in FEATS if f.endswith(("_nuevo", "_nueva"))]
s = df[novedad].sum(axis=1)
ap_base = average_precision_score(y, s)
seccion("Línea base: suma de las variables de novedad")
tabla(pd.DataFrame([
    {"Método": "Azar", "AP": y.mean(), "AUC": 0.5, "Veces el azar": 1.0},
    {"Método": "Suma de novedades", "AP": ap_base,
     "AUC": roc_auc_score(y, s), "Veces el azar": ap_base / y.mean()},
]))

