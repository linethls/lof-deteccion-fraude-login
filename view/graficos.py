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

# ---------- Datos ----------
y = np.load(PROC / "final_test_y.npy")
nombres = ["Línea base\n(suma novedades)",
           "LOF todas las variables\n(k=20)",
           "LOF sin seg/login_ok, x3\n(k=50)"]
scores = [np.load(PROC / f"final_test_{i}.npy") for i in range(3)]
n_at = int(y.sum())
n_tot = len(y)
azar_pr = y.mean()

# Paleta apta para daltónicos (gris = referencia, azul/naranja = modelos)
colores = ["#7f7f7f", "#0072B2", "#D55E00"]

# ---------- Métricas ----------
auc = [roc_auc_score(y, s) for s in scores]
ap = [average_precision_score(y, s) for s in scores]
mejor = int(np.argmax(ap))

def recall_en_fpr(s, objetivo):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(objetivo, fpr, tpr))

fprs_obj = [0.01, 0.05]
rec = {f: [recall_en_fpr(s, f) for s in scores] for f in fprs_obj}

# ---------- Estilo general ----------
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white",
})

def etiqueta(i):
    return nombres[i].replace("\n", " ")

# ---------- Funciones de dibujo ----------
def dibujar_roc(ax):
    for i, s in enumerate(scores):
        fpr, tpr, _ = roc_curve(y, s)
        lw = 3 if i == mejor else 2
        ax.plot(fpr, tpr, color=colores[i], lw=lw,
                label=f"{etiqueta(i)}  (AUC = {auc[i]:.2f})", zorder=3 + (i == mejor))
        if i == mejor:
            ax.fill_between(fpr, tpr, alpha=0.12, color=colores[i])
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Azar (AUC = 0.50)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Falsas alarmas (tasa de falsos positivos)")
    ax.set_ylabel("Ataques detectados (recall)")
    ax.set_title("Curva ROC\nmás arriba y a la izquierda = mejor")
    ax.legend(loc="lower right", fontsize=8, frameon=True)

def dibujar_pr(ax):
    for i, s in enumerate(scores):
        p, r, _ = precision_recall_curve(y, s)
        lw = 3 if i == mejor else 2
        ax.plot(r, p, color=colores[i], lw=lw,
                label=f"{etiqueta(i)}  (AP = {ap[i]:.2f})", zorder=3 + (i == mejor))
        if i == mejor:
            ax.fill_between(r, p, alpha=0.12, color=colores[i])
    ax.axhline(azar_pr, color="k", ls="--", lw=1,
               label=f"Azar (precisión = {azar_pr:.2f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Ataques detectados (recall)")
    ax.set_ylabel("Precisión (de las alertas, cuántas son ataques reales)")
    ax.set_title("Curva precisión-recall\nmás arriba y a la derecha = mejor")
    ax.legend(loc="upper right", fontsize=8, frameon=True)

def barras(ax, valores, titulo, ylabel, referencia=None, ref_txt=""):
    x = np.arange(len(valores))
    b = ax.bar(x, valores, color=colores, width=0.6, edgecolor="white")
    for rect, v in zip(b, valores):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.015, f"{v:.2f}",
                ha="center", va="bottom", fontweight="bold")
    if referencia is not None:
        ax.axhline(referencia, color="k", ls="--", lw=1)
        ax.text(len(valores) - 0.5, referencia + 0.015, ref_txt,
                ha="right", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Línea\nbase", "LOF\ntodas vars", "LOF\nsin seg/login"],
                       fontsize=9)
    ax.set_ylim(0, 1.1); ax.set_ylabel(ylabel); ax.set_title(titulo)
    ax.grid(axis="x", visible=False)

def dibujar_resumen(ax):
    """Barras agrupadas: recall con 1% y 5% de falsas alarmas."""
    ancho = 0.35
    x = np.arange(len(scores))
    b1 = ax.bar(x - ancho / 2, rec[0.01], ancho, label="Con 1 % de falsas alarmas",
                color=colores, alpha=0.55, edgecolor="white")
    b2 = ax.bar(x + ancho / 2, rec[0.05], ancho, label="Con 5 % de falsas alarmas",
                color=colores, edgecolor="white")
    for bars in (b1, b2):
        for rect in bars:
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2, h + 0.015, f"{h:.0%}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Línea\nbase", "LOF\ntodas vars", "LOF\nsin seg/login"],
                       fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("% de ataques detectados")
    ax.set_title("¿Cuántos ataques detecto tolerando\npocas falsas alarmas?")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="x", visible=False)

# ---------- Gráficos individuales ----------
for nombre_archivo, fn, size in [("roc_final.png", dibujar_roc, (7, 6)),
                                 ("pr_final.png", dibujar_pr, (7, 6))]:
    fig, ax = plt.subplots(figsize=size)
    fn(ax)
    fig.suptitle(f"Evaluación en test: {n_at} ataques de {n_tot} eventos",
                 fontsize=10, color="gray", y=0.99)
    fig.tight_layout()
    fig.savefig(RES / nombre_archivo, dpi=200)
    plt.close(fig)

# ---------- Panel resumen ----------
fig, axs = plt.subplots(2, 2, figsize=(14, 11))
dibujar_roc(axs[0, 0])
dibujar_pr(axs[0, 1])
barras(axs[1, 0], auc, "AUC-ROC (1.0 = perfecto)", "AUC",
       referencia=0.5, ref_txt="Azar = 0.50")
dibujar_resumen(axs[1, 1])
fig.suptitle(f"Comparación de modelos de detección de anomalías, conjunto de test "
             f"({n_at} ataques de {n_tot} eventos, {azar_pr:.1%} del total)",
             fontsize=15, fontweight="bold")
fig.text(0.5, 0.005,
         f"Mejor modelo según AP: {etiqueta(mejor)}",
         ha="center", fontsize=11, color=colores[mejor], fontweight="bold")
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig(RES / "panel_resumen.png", dpi=200)
plt.close(fig)

print("Gráficos guardados en", RES)
print("Archivos: roc_final.png, pr_final.png, panel_resumen.png")