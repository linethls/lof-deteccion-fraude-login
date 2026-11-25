import sys
import numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
from formato import titulo, seccion, nota, tabla, miles

PROC = RAIZ / "datos" / "procesado"
MODELOS = RAIZ / "modelos"
MODELOS.mkdir(parents=True, exist_ok=True)



df = pd.read_csv(PROC / "features.csv")
FEATS = [c for c in df.columns if c not in ("y", "usuario", "fecha")]
X, y, g = df[FEATS].values, df["y"].values, df["usuario"].values
assert np.isfinite(X).all(), "Hay NaN o inf en las variables"

seccion("Datos de entrada (features.csv)")
tabla(pd.DataFrame([
    {"Dato": "Accesos (filas)", "Valor": miles(len(df))},
    {"Dato": "Usuarios distintos", "Valor": miles(len(set(g)))},
    {"Dato": "Variables usadas", "Valor": len(FEATS)},
    {"Dato": "Tomas de cuenta", "Valor": miles(y.sum())},
    {"Dato": "Proporción de ataques", "Valor": f"{100 * y.mean():.2f} %"},
]))


def partir(idx, n_splits):
    sg = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    a, b = next(sg.split(X[idx], y[idx], g[idx]))
    return idx[a], idx[b]


idx = np.arange(len(df))
i_trval, i_te = partir(idx, 5)      # ~20 % prueba
i_tr, i_va = partir(i_trval, 4)     # ~20 % validación

assert not (set(g[i_tr]) & set(g[i_te]))
assert not (set(g[i_tr]) & set(g[i_va]))
assert not (set(g[i_va]) & set(g[i_te]))

scaler = StandardScaler().fit(X[i_tr])     # solo con entrenamiento
bloques = [("train", "Entrenamiento", i_tr), ("val", "Validación", i_va),
           ("test", "Prueba", i_te)]
filas = []
for clave, nombre, ix in bloques:
    np.save(PROC / f"X_{clave}_s.npy", scaler.transform(X[ix]))
    np.save(PROC / f"y_{clave}.npy", y[ix])
    filas.append({"Bloque": nombre, "Accesos": miles(len(ix)),
                  "Usuarios": miles(len(set(g[ix]))),
                  "Ataques": int(y[ix].sum()),
                  "% ataques": f"{100 * y[ix].mean():.2f}"})
joblib.dump(scaler, MODELOS / "scaler.joblib")

seccion("Resultado de la partición")
tabla(pd.DataFrame(filas))
print("\nVerificación: ningún usuario aparece en dos bloques ........ OK")
print("Verificación: sin valores NaN ni infinitos en las variables . OK")
print(f"Archivos guardados en: {PROC}")


if min(f["Ataques"] for f in filas) < 10:
    puntos.append("ATENCIÓN: algún bloque tiene menos de 10 ataques; las "
                  "métricas serán muy ruidosas.")
