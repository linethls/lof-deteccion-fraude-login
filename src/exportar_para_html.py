"""Genera view/data/data.js para el demo HTML del LOF.

El modelo del demo replica el que se evaluó en evaluar_final.py:
- variables: las 6 de "sin_cont" (sin seg_desde_ultimo ni login_ok)
- se estandariza con el scaler de preparar.py y DESPUÉS se multiplican las
  novedades por 3 (antes se pesaba primero y el escalador anulaba el x3)
- LOF con k = 50, ajustado solo con accesos normales de entrenamiento
- umbral fijado en VALIDACIÓN (percentil 95), como en la Tabla 2

Uso: python view/exportar_para_html.py
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import LocalOutlierFactor

RAIZ = Path(__file__).resolve().parents[1]
PROC = RAIZ / "datos" / "procesado"
MODELOS = RAIZ / "modelos"
SALIDA_JS = RAIZ / "view" / "data" / "data.js"

K = 50
N_JS = 2000          # puntos de entrenamiento que viajan al navegador
N_PTS = 400          # puntos del gráfico de vecinos
N_USUARIOS = 200     # usuarios del selector del demo

# Países que el HTML sabe dibujar (misma lista que PAISES_DB en el JS)
PAISES_DEMO = {
    "RU", "BO", "NO", "DE", "BR", "US", "GB", "FR", "CA", "IN",
    "UA", "PL", "NL", "RO", "CH", "TH", "DK",
}

rng = np.random.default_rng(42)

# 1. Variables (mismo orden que preparar.py)
FEATS = [
    c for c in pd.read_csv(PROC / "features.csv", nrows=0).columns
    if c not in ("y", "usuario", "fecha")
]
sin_cont = [
    i for i, f in enumerate(FEATS)
    if f not in ("seg_desde_ultimo", "login_ok")
]
feats_sub = [FEATS[i] for i in sin_cont]
pesos = np.array([
    3.0 if f.endswith(("_nuevo", "_nueva")) else 1.0
    for f in feats_sub
])

# 2. Datos ya estandarizados por preparar.py (scaler ajustado con entrenamiento)
scaler = joblib.load(MODELOS / "scaler.joblib")
X_tr = np.load(PROC / "X_train_s.npy")
y_tr = np.load(PROC / "y_train.npy")
X_va = np.load(PROC / "X_val_s.npy")

normales = X_tr[y_tr == 0]
X_fit = normales[:, sin_cont] * pesos                 # estandarizar y luego pesar
raw = normales * scaler.scale_ + scaler.mean_         # valores originales (gráfico)

# 3. Submuestra para el navegador y LOF sobre ESA submuestra,
#    para que el umbral sea comparable con lo que calcula el JS
n_js = min(N_JS, len(X_fit))
idx_js = rng.choice(len(X_fit), n_js, replace=False)
X_js, raw_js = X_fit[idx_js], raw[idx_js]

lof = LocalOutlierFactor(n_neighbors=K, novelty=True, n_jobs=-1).fit(X_js)
s_va = -lof.score_samples(X_va[:, sin_cont] * pesos)
THR = float(np.percentile(s_va, 95))                  # fijado en validación

# 4. Puntos del gráfico de vecinos: hora y número de novedades (0 a 4)
col = {f: i for i, f in enumerate(FEATS)}
nov_cols = [col[f] for f in FEATS if f.endswith(("_nuevo", "_nueva"))]
train_pts = []
for i in rng.choice(n_js, min(N_PTS, n_js), replace=False):
    ang = np.arctan2(raw_js[i, col["hora_sin"]], raw_js[i, col["hora_cos"]])
    hora = (ang * 24 / (2 * np.pi)) % 24
    train_pts.append([round(float(hora), 2), int(round(raw_js[i, nov_cols].sum()))])

# 5. Usuarios reales para el selector: uno por usuario, solo normales
cols = [
    "Login Timestamp", "User ID", "Country", "Device Type",
    "Browser Name and Version", "Is Account Takeover",
]
# keep_default_na=False evita que el país "NA" (Namibia) se lea como vacío
df_raw = pd.read_csv(
    RAIZ / "datos" / "muestra.csv",
    usecols=cols,
    dtype=str,
    keep_default_na=False,
)
atacados = set(df_raw.loc[df_raw["Is Account Takeover"] == "True", "User ID"])
candidatos = df_raw[
    ~df_raw["User ID"].isin(atacados) & df_raw["Country"].isin(PAISES_DEMO)
]
candidatos = candidatos.drop_duplicates("User ID")
muestra = candidatos.sample(n=min(N_USUARIOS, len(candidatos)), random_state=42)

ts = pd.to_datetime(muestra["Login Timestamp"])
usuarios_reales = [
    {
        "id": r["User ID"],
        "pais": r["Country"],
        "disp": r["Device Type"] + " (" + r["Browser Name and Version"][:15] + ")",
        "hora": round(float(t.hour + t.minute / 60), 2),
    }
    for (_, r), t in zip(muestra.iterrows(), ts)
]

# 6. Ataques REALES (toma de cuenta confirmada) para el simulador.
#    Se reproducen con las variables que calculó variables.py, así el demo
#    puntúa exactamente lo que vio el modelo.
feat_df = pd.read_csv(PROC / "features.csv", dtype={"usuario": str})
full = pd.read_csv(
    RAIZ / "datos" / "muestra.csv",
    usecols=cols,
    dtype=str,
    keep_default_na=False,
)
full["Login Timestamp"] = pd.to_datetime(full["Login Timestamp"])
# Mismo orden que usó variables.py, para que cada fila coincida con features.csv
full = full.sort_values(["User ID", "Login Timestamp"]).reset_index(drop=True)
assert (full["User ID"].values == feat_df["usuario"].values).all(), \
    "muestra.csv y features.csv no están alineados"

full["y"] = feat_df["y"].values
NOV = ["pais_nuevo", "asn_nuevo", "ip_nueva", "dispositivo_nuevo"]
for c in NOV:
    full[c] = feat_df[c].values

normal = full[full["y"] == 0]
primero = normal.groupby("User ID").first()           # dispositivo y hora habituales
hogar = normal.groupby("User ID")["Country"].agg(lambda s: s.mode().iloc[0])

ataques_reales, descartados = [], 0
for _, r in full[full["y"] == 1].iterrows():
    u = r["User ID"]
    if u not in hogar.index:                          # usuario sin accesos normales
        descartados += 1
        continue
    h_pais, a_pais = hogar[u], r["Country"]
    if h_pais not in PAISES_DEMO or a_pais not in PAISES_DEMO:
        descartados += 1
        continue                                      # el mapa no sabe dibujarlo
    if int(r["pais_nuevo"]) != int(a_pais != h_pais): # el demo deduce país nuevo así
        descartados += 1
        continue
    t, t0 = r["Login Timestamp"], primero.loc[u, "Login Timestamp"]
    ataques_reales.append({
        "id": u,
        "home": h_pais,
        "pais": a_pais,
        "disp": (
            primero.loc[u, "Device Type"]
            + " ("
            + primero.loc[u, "Browser Name and Version"][:15]
            + ")"
        ),
        "hora": round(float(t.hour + t.minute / 60), 2),
        "hora_home": round(float(t0.hour + t0.minute / 60), 2),
        "pais_nuevo": int(r["pais_nuevo"]),
        "asn_nuevo": int(r["asn_nuevo"]),
        "ip_nueva": int(r["ip_nueva"]),
        "dispositivo_nuevo": int(r["dispositivo_nuevo"]),
    })
rng.shuffle(ataques_reales)

# 7. Exportar
datos = {
    "scaler_mean": scaler.mean_[sin_cont].tolist(),
    "scaler_scale": scaler.scale_[sin_cont].tolist(),
    "features_sub": feats_sub,
    "pesos": pesos.tolist(),
    "X_train_s": X_js.round(4).tolist(),
    "k": K,
    "THR": THR,
    "trainPts": train_pts,
    "n_train_total": int(len(X_fit)),
    "usuarios_reales": usuarios_reales,
    "ataques_reales": ataques_reales,
}
js = (
    "// Generado automáticamente por exportar_para_html.py\n"
    "window.LOF_DATA = " + json.dumps(datos) + ";\n"
)

SALIDA_JS.parent.mkdir(parents=True, exist_ok=True)
SALIDA_JS.write_text(js, encoding="utf-8")

print(f"Archivo generado: {SALIDA_JS}")
print(f"  Puntos de ajuste en el navegador: {n_js} de {len(X_fit)} normales")
print(f"  Umbral (percentil 95 en validación): {THR:.3f}")
print(f"  Usuarios reales en el selector: {len(usuarios_reales)}")
print(
    f"  Ataques reales para el simulador: {len(ataques_reales)} "
    f"({descartados} descartados por país o historial)"
)