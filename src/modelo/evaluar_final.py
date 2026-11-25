import sys
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import average_precision_score, roc_auc_score

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
from formato import titulo, seccion, nota, tabla, miles

PROC = RAIZ / "datos" / "procesado"
RES = RAIZ / "resultados"
RES.mkdir(parents=True, exist_ok=True)



df = pd.read_csv(PROC / "features.csv")
FEATS = [c for c in df.columns if c not in ("y", "usuario", "fecha")]
X_tr = np.load(PROC / "X_train_s.npy"); y_tr = np.load(PROC / "y_train.npy")
X_va = np.load(PROC / "X_val_s.npy");   y_va = np.load(PROC / "y_val.npy")
X_te = np.load(PROC / "X_test_s.npy");  y_te = np.load(PROC / "y_test.npy")

seccion("Bloque de prueba")
tabla(pd.DataFrame([
    {"Dato": "Accesos de prueba", "Valor": miles(len(y_te))},
    {"Dato": "Tomas de cuenta", "Valor": int(y_te.sum())},
    {"Dato": "Azar (AP)", "Valor": f"{y_te.mean():.3f}"},
    {"Dato": "Azar (AUC)", "Valor": "0.500"},
]))

nov = [i for i, f in enumerate(FEATS) if f.endswith(("_nuevo", "_nueva"))]
sin_cont = [i for i, f in enumerate(FEATS) if f not in ("seg_desde_ultimo", "login_ok")]
pesos = np.array([3.0 if FEATS[i].endswith(("_nuevo", "_nueva")) else 1.0
                  for i in sin_cont])


def lof_scores(cols, k, pesos=None):
    A, V, T = X_tr[:, cols], X_va[:, cols], X_te[:, cols]
    if pesos is not None:
        A, V, T = A * pesos, V * pesos, T * pesos
    lof = LocalOutlierFactor(n_neighbors=k, novelty=True, n_jobs=-1).fit(A[y_tr == 0])
    return -lof.score_samples(V), -lof.score_samples(T)


candidatos = {
    "Línea base (suma novedades)": (X_va[:, nov].sum(axis=1), X_te[:, nov].sum(axis=1)),
    "LOF todas las variables (k=20)": lof_scores(list(range(len(FEATS))), 20),
    "LOF sin seg/login_ok, x3 (k=50)": lof_scores(sin_cont, 50, pesos),
}

print("\nCalculando intervalos de confianza (1000 remuestreos por método)...")
rng = np.random.default_rng(42)
filas, detectados, intervalos = [], {}, {}
for nombre, (s_va, s_te) in candidatos.items():
    ap = average_precision_score(y_te, s_te)
    auc = roc_auc_score(y_te, s_te)

    umbral = np.percentile(s_va, 95)          # fijado en VALIDACIÓN
    pred = (s_te >= umbral).astype(int)
    vp = int(((pred == 1) & (y_te == 1)).sum())
    recall = vp / y_te.sum()
    precision = vp / max(pred.sum(), 1)

    aps, aucs = [], []
    for _ in range(1000):
        i = rng.integers(0, len(y_te), len(y_te))
        if y_te[i].sum() == 0:
            continue
        aps.append(average_precision_score(y_te[i], s_te[i]))
        aucs.append(roc_auc_score(y_te[i], s_te[i]))
    intervalos[nombre] = (np.percentile(aps, 2.5), np.percentile(aps, 97.5))
    detectados[nombre] = vp
    filas.append({
        "método": nombre,
        "AP": round(ap, 3),
        "AP IC95%": f"{np.percentile(aps, 2.5):.3f}-{np.percentile(aps, 97.5):.3f}",
        "AUC": round(auc, 3),
        "AUC IC95%": f"{np.percentile(aucs, 2.5):.3f}-{np.percentile(aucs, 97.5):.3f}",
        "marcados %": round(100 * pred.mean(), 1),
        "recall": round(recall, 3),
        "precisión": round(precision, 3),
    })

res = pd.DataFrame(filas)
res.to_csv(RES / "resultados_finales_test.csv", index=False)

seccion("Tabla 1. Capacidad de ordenar los accesos (no depende del umbral)")
t1 = pd.DataFrame({
    "Método": res["método"],
    "AP [IC 95 %]": [f"{a:.3f} [{i}]" for a, i in zip(res["AP"], res["AP IC95%"])],
    "AUC [IC 95 %]": [f"{a:.3f} [{i}]" for a, i in zip(res["AUC"], res["AUC IC95%"])],
})
tabla(t1)
print(f"Azar: AP = {y_te.mean():.3f} | AUC = 0.500")

seccion("Tabla 2. Alertas con el umbral fijado en validación (5 % más alto)")
t2 = pd.DataFrame({
    "Método": res["método"],
    "Marcados %": res["marcados %"],
    "Detectados": [f"{detectados[n]} de {int(y_te.sum())}" for n in res["método"]],
    "Recall": res["recall"],
    "Precisión": res["precisión"],
})
tabla(t2, decimales={"Marcados %": 1})

orden = sorted(intervalos, key=lambda n: -res.loc[res["método"] == n, "AP"].iloc[0])
a, b = orden[0], orden[1]
solapan = intervalos[a][0] <= intervalos[b][1] and intervalos[b][0] <= intervalos[a][1]
conclusion = (
    f"Los intervalos de AP de '{a}' y '{b}' se solapan: con {int(y_te.sum())} "
    "ataques no se puede declarar un método ganador."
    if solapan else
    f"Los intervalos de AP de '{a}' y '{b}' no se solapan: la diferencia es consistente.")



for i, (n, (sv, st)) in enumerate(candidatos.items()):
    np.save(PROC / f"final_test_{i}.npy", st)
np.save(PROC / "final_test_y.npy", y_te)
print(f"Archivos guardados: resultados/resultados_finales_test.csv y puntajes para los gráficos")