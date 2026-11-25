import sys
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import average_precision_score, roc_auc_score

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
from formato import titulo, seccion, nota, tabla, miles

PROC = RAIZ / "datos" / "procesado"


df = pd.read_csv(PROC / "features.csv")
FEATS = [c for c in df.columns if c not in ("y", "usuario", "fecha")]
X_tr = np.load(PROC / "X_train_s.npy"); y_tr = np.load(PROC / "y_train.npy")
X_va = np.load(PROC / "X_val_s.npy");   y_va = np.load(PROC / "y_val.npy")
azar = y_va.mean()

nov = [i for i, f in enumerate(FEATS) if f.endswith(("_nuevo", "_nueva"))]
base = X_va[:, nov].sum(axis=1)
filas = [{"Variante": "Línea base (suma de novedades)", "k": "-",
          "AP": average_precision_score(y_va, base),
          "AUC": roc_auc_score(y_va, base),
          "Veces el azar": average_precision_score(y_va, base) / azar}]


def probar(nombre, cols, pesos=None):
    A, B = X_tr[:, cols], X_va[:, cols]
    if pesos is not None:
        A, B = A * pesos, B * pesos
    A = A[y_tr == 0]
    for k in [20, 50, 100]:
        lof = LocalOutlierFactor(n_neighbors=k, novelty=True, n_jobs=-1).fit(A)
        s = -lof.score_samples(B)
        ap = average_precision_score(y_va, s)
        filas.append({"Variante": nombre, "k": k, "AP": ap,
                      "AUC": roc_auc_score(y_va, s), "Veces el azar": ap / azar})


todas = list(range(len(FEATS)))
sin_cont = [i for i, f in enumerate(FEATS) if f not in ("seg_desde_ultimo", "login_ok")]
pesos = np.array([3.0 if FEATS[i].endswith(("_nuevo", "_nueva")) else 1.0
                  for i in sin_cont])

print(f"Entrenando 9 modelos LOF (3 variantes x 3 valores de k)...")
probar("LOF con todas las variables", todas)
probar("LOF sin seg_desde_ultimo ni login_ok", sin_cont)
probar("LOF sin esas dos y novedades x3", sin_cont, pesos)

seccion("Resultados en validación")
tabla(pd.DataFrame(filas), izquierda=("Variante",))
lof_filas = [f for f in filas[1:]]
mejor = max(lof_filas, key=lambda f: f["AP"])
print(f"\nMejor variante de LOF: '{mejor['Variante']}' con k = {mejor['k']} "
      f"(AP = {mejor['AP']:.3f}, {mejor['Veces el azar']:.1f} veces el azar).")
print(f"Azar (AP) = {azar:.3f}")

