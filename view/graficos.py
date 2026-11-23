import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (roc_curve, precision_recall_curve,
                             roc_auc_score, average_precision_score)

RAIZ = Path(__file__).resolve().parents[1]
PROC = RAIZ / "datos" / "procesado"
RES = RAIZ / "resultados"
RES.mkdir(parents=True, exist_ok=True)

y = np.load(PROC / "final_test_y.npy")
nombres = ["Línea base (suma novedades)", "LOF todas las variables (k=20)",
           "LOF sin seg/login_ok, x3 (k=50)"]
scores = [np.load(PROC / f"final_test_{i}.npy") for i in range(3)]
n_at = int(y.sum())

plt.figure(figsize=(6, 5))
for n, s in zip(nombres, scores):
    fpr, tpr, _ = roc_curve(y, s)
    plt.plot(fpr, tpr, label=f"{n} (AUC={roc_auc_score(y, s):.2f})")
plt.plot([0, 1], [0, 1], "k--", label="Azar")
plt.xlabel("Tasa de falsos positivos"); plt.ylabel("Tasa de verdaderos positivos")
plt.title(f"Curva ROC en test ({n_at} ataques)")
plt.legend(fontsize=7); plt.tight_layout()
plt.savefig(RES / "roc_final.png", dpi=150); plt.close()

plt.figure(figsize=(6, 5))
for n, s in zip(nombres, scores):
    p, r, _ = precision_recall_curve(y, s)
    plt.plot(r, p, label=f"{n} (AP={average_precision_score(y, s):.2f})")
plt.axhline(y.mean(), color="k", linestyle="--", label="Azar")
plt.xlabel("Recall"); plt.ylabel("Precisión")
plt.title(f"Curva precisión-recall en test ({n_at} ataques)")
plt.legend(fontsize=7); plt.tight_layout()
plt.savefig(RES / "pr_final.png", dpi=150); plt.close()
print("Gráficos guardados en", RES)