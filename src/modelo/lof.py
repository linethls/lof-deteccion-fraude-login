import sys, json, time
import numpy as np, pandas as pd, joblib
from pathlib import Path
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import average_precision_score

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
from formato import titulo, seccion, nota, tabla, miles

PROC = RAIZ / "datos" / "procesado"
MODELOS = RAIZ / "modelos"
MODELOS.mkdir(parents=True, exist_ok=True)

KS = [10, 20, 50, 100]
PCTS = [0.5, 1, 2, 5]



X_tr = np.load(PROC / "X_train_s.npy"); y_tr = np.load(PROC / "y_train.npy")
X_va = np.load(PROC / "X_val_s.npy");   y_va = np.load(PROC / "y_val.npy")
X_te = np.load(PROC / "X_test_s.npy")
azar = y_va.mean()

ESCENARIOS = {
    "todos": ("Todos (no supervisado)", X_tr),
    "normales": ("Solo normales (semisup.)", X_tr[y_tr == 0]),
}

seccion("Datos de ajuste de cada escenario")
tabla(pd.DataFrame([
    {"Escenario": desc, "Filas": miles(len(X)),
     "Filas únicas": miles(len(np.unique(X, axis=0)))}
    for desc, X in ESCENARIOS.values()]))

print("\nEntrenando 8 modelos (2 escenarios x 4 valores de k)...")
filas, resumen = [], {}
for nombre, (desc, X_fit) in ESCENARIOS.items():
    resumen[nombre] = {}
    for k in KS:
        t = time.time()
        lof = LocalOutlierFactor(n_neighbors=k, novelty=True, n_jobs=-1).fit(X_fit)
        s_va = -lof.score_samples(X_va)
        s_te = -lof.score_samples(X_te)
        assert np.isfinite(s_va).all() and np.isfinite(s_te).all(), "scores no finitos"

        ap = average_precision_score(y_va, s_va)
        resumen[nombre][k] = ap
        umbrales = {str(p): float(np.percentile(s_va, 100 - p)) for p in PCTS}
        joblib.dump(lof, MODELOS / f"lof_{nombre}_k{k}.joblib")
        (MODELOS / f"umbrales_{nombre}_k{k}.json").write_text(json.dumps(umbrales))
        np.save(PROC / f"scores_val_{nombre}_k{k}.npy", s_va)
        np.save(PROC / f"scores_test_{nombre}_k{k}.npy", s_te)
        filas.append({"Escenario": desc, "k": k, "AP validación": ap,
                      "Veces el azar": ap / azar, "Segundos": time.time() - t})

mejor = max(((n, k) for n in resumen for k in KS), key=lambda t: resumen[t[0]][t[1]])
(MODELOS / "mejor.json").write_text(json.dumps(
    {"escenario": mejor[0], "k": mejor[1], "ap_val": resumen}))

seccion("Average precision (AP) en validación")
tabla(pd.DataFrame(filas), decimales={"Segundos": 1})
print(f"\nMejor combinación según validación: escenario '{mejor[0]}', k = {mejor[1]}")
print(f"Azar (AP) = proporción de ataques en validación = {azar:.3f}")


