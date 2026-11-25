import sys, time
import numpy as np, pandas as pd
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RAIZ / "src"))
from formato import titulo, seccion, nota, tabla, miles

CSV = RAIZ / "datos" / "dataset" / "rba-dataset.csv"
SALIDA = RAIZ / "datos" / "muestra.csv"
TMP = SALIDA.with_suffix(".tmp")
N_NORMALES = 2000
BLOQUE = 1_000_000
rng = np.random.default_rng(42)



if not CSV.exists():
    raise SystemExit(f"No encuentro el dataset en {CSV}")
SALIDA.parent.mkdir(parents=True, exist_ok=True)

# ---------- Pasada 1: usuarios y usuarios con toma de cuenta ----------
t1 = time.time()
usuarios, ato, filas_p1 = set(), set(), 0
for ch in pd.read_csv(CSV, usecols=["User ID", "Is Account Takeover"],
                      chunksize=BLOQUE):
    usuarios.update(ch["User ID"].unique())
    ato.update(ch.loc[ch["Is Account Takeover"] == True, "User ID"].unique())
    filas_p1 += len(ch)
    print(f"  Pasada 1: {miles(filas_p1)} filas leídas", end="\r", flush=True)
print(f"  Pasada 1: {miles(filas_p1)} filas leídas ... listo")
seg1 = time.time() - t1

normales = np.array(sorted(usuarios - ato))
n_normales = min(N_NORMALES, len(normales))
elegidos = list(set(rng.choice(normales, n_normales, replace=False).tolist()) | ato)

# ---------- Pasada 2: guardar TODOS los accesos de los usuarios elegidos ----------
t2 = time.time()
primera, filas_p2, total, n_ato = True, 0, 0, 0
for ch in pd.read_csv(CSV, chunksize=BLOQUE):
    parte = ch[ch["User ID"].isin(elegidos)]
    parte.to_csv(TMP, mode="w" if primera else "a", header=primera, index=False)
    primera = False
    filas_p2 += len(ch)
    total += len(parte)
    n_ato += int(parte["Is Account Takeover"].sum())
    print(f"  Pasada 2: {miles(filas_p2)} filas leídas", end="\r", flush=True)
print(f"  Pasada 2: {miles(filas_p2)} filas leídas ... listo")
TMP.replace(SALIDA)          # el archivo final solo existe si todo terminó bien
seg2 = time.time() - t2

# ---------- Resultados ----------
seccion("Tabla 1. Qué hizo cada pasada")
tabla(pd.DataFrame([
    {"Pasada": "1. Identificar usuarios", "Filas leídas": miles(filas_p1),
     "Resultado": f"{miles(len(usuarios))} usuarios, {miles(len(ato))} con toma de cuenta",
     "Segundos": seg1},
    {"Pasada": "2. Guardar accesos", "Filas leídas": miles(filas_p2),
     "Resultado": f"{miles(total)} accesos guardados", "Segundos": seg2},
]), decimales={"Segundos": 0}, izquierda=("Pasada", "Resultado"))

seccion("Tabla 2. Resumen de la muestra")
tabla(pd.DataFrame([
    {"Concepto": "Usuarios en el dataset", "Valor": miles(len(usuarios))},
    {"Concepto": "Usuarios con toma de cuenta (se conservan todos)", "Valor": miles(len(ato))},
    {"Concepto": "Usuarios normales elegidos al azar", "Valor": miles(n_normales)},
    {"Concepto": "Usuarios en la muestra", "Valor": miles(len(elegidos))},
    {"Concepto": "Accesos en la muestra", "Valor": miles(total)},
    {"Concepto": "Tomas de cuenta en la muestra", "Valor": f"{miles(n_ato)} ({100 * n_ato / total:.1f} %)"},
    {"Concepto": "Archivo generado", "Valor": str(SALIDA.relative_to(RAIZ))},
]))

