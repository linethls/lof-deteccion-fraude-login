import numpy as np, joblib, json, time
from pathlib import Path
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import average_precision_score

RAIZ = Path(__file__).resolve().parents[2]
PROC = RAIZ / "datos" / "procesado"
MODELOS = RAIZ / "modelos"
MODELOS.mkdir(parents=True, exist_ok=True)

KS = [10, 20, 50, 100]
PCTS = [0.5, 1, 2, 5]

X_tr = np.load(PROC / "X_train_s.npy"); y_tr = np.load(PROC / "y_train.npy")
X_va = np.load(PROC / "X_val_s.npy");   y_va = np.load(PROC / "y_val.npy")
X_te = np.load(PROC / "X_test_s.npy")

ESCENARIOS = {
    "todos": X_tr,                 # sin usar etiquetas (incluye ataques)
    "normales": X_tr[y_tr == 0],   # solo logins normales
}

resumen = {}
for nombre, X_fit in ESCENARIOS.items():
    print(f"\n--- Escenario '{nombre}': {len(X_fit)} filas, "
          f"{len(np.unique(X_fit, axis=0))} únicas ---")
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
        print(f"k={k}: AP validación = {ap:.3f} | {time.time() - t:.1f} s")

mejor = max(((n, k) for n in resumen for k in KS), key=lambda t: resumen[t[0]][t[1]])
(MODELOS / "mejor.json").write_text(json.dumps(
    {"escenario": mejor[0], "k": mejor[1], "ap_val": resumen}))
print("\nMejor (según validación):", mejor)