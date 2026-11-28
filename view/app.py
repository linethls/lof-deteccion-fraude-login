import joblib
import numpy as np
import streamlit as st
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
MODELOS = RAIZ / "modelos"
PROC = RAIZ / "datos" / "procesado"

st.set_page_config(page_title="Detector de logins sospechosos", page_icon="🔐")
st.title("🔐 Detector de inicios de sesión sospechosos")
st.caption("Modelo: Local Outlier Factor (LOF, k=50, 6 variables ponderadas x3, ajustado con accesos normales)")

# ---- Comprobar que existen el modelo, el escalador y los puntajes de validación ----
modelo_path = MODELOS / "lof_mejorado_k50.joblib"
scaler_path = MODELOS / "scaler.joblib"
scores_val_path = PROC / "scores_val_lof_mejorado_k50.npy"

if not (modelo_path.exists() and scaler_path.exists() and scores_val_path.exists()):
    st.error("No encuentro los archivos del modelo mejorado. Ejecuta primero preparar.py y evaluar_final.py.")
    st.stop()

pct = st.sidebar.selectbox(
    "Marcar como sospechoso el % más alto (en validación)",
    [0.5, 1, 2, 5],
    index=3,  # por defecto 5 %, igual que en la Tabla 2 del informe
)

scaler = joblib.load(scaler_path)
lof = joblib.load(modelo_path)
umbral = float(np.percentile(np.load(scores_val_path), 100 - pct))

# Índices de las 6 variables en scaler (excluye login_ok [2] y seg_desde_ultimo [7])
IDX_6 = [0, 1, 3, 4, 5, 6]
PESOS_6 = np.array([1.0, 1.0, 3.0, 3.0, 3.0, 3.0])

# ---- Casos de prueba para la demostración ----
CASOS = {
    "Login normal": dict(hora=14, pais=0, asn=0, ip=0, disp=0),
    "3 a. m. desde un país nuevo": dict(hora=3, pais=1, asn=1, ip=1, disp=0),
    "Todo nuevo de madrugada": dict(hora=4, pais=1, asn=1, ip=1, disp=1),
    "Personalizado": dict(hora=12, pais=0, asn=0, ip=0, disp=0),
}
caso = st.selectbox("Caso de prueba", list(CASOS))
c = CASOS[caso]

col1, col2 = st.columns(2)
with col1:
    hora = st.slider("Hora del acceso", 0, 23, c["hora"], key=f"{caso}_hora")
    pais = st.checkbox("País nuevo para este usuario", bool(c["pais"]), key=f"{caso}_pais")
    asn = st.checkbox("Red (ASN) nueva", bool(c["asn"]), key=f"{caso}_asn")
with col2:
    ip = st.checkbox("IP nueva", bool(c["ip"]), key=f"{caso}_ip")
    disp = st.checkbox("Dispositivo nuevo", bool(c["disp"]), key=f"{caso}_disp")

if st.button("Evaluar login", type="primary"):
    # Mismo orden que sin_cont en evaluar_final.py:
    # ["hora_sin", "hora_cos", "pais_nuevo", "asn_nuevo", "ip_nueva", "dispositivo_nuevo"]
    x_raw = np.array([[
        np.sin(2 * np.pi * hora / 24),
        np.cos(2 * np.pi * hora / 24),
        int(pais),
        int(asn),
        int(ip),
        int(disp),
    ]])

    # Estandarizar con el scaler de entrenamiento y DESPUÉS aplicar los pesos x3
    x_std = (x_raw - scaler.mean_[IDX_6]) / scaler.scale_[IDX_6]
    x_final = x_std * PESOS_6

    puntaje = float(-lof.score_samples(x_final)[0])

    if puntaje >= umbral:
        st.error("⚠️ SOSPECHOSO: conviene pedir verificación adicional (MFA)")
    else:
        st.success("✅ NORMAL: acceso habitual")

    st.metric("Puntaje LOF", f"{puntaje:.2f}", f"umbral {umbral:.2f}", delta_color="off")
    st.caption(
        "Cerca de 1 es normal. Mientras más alto, más aislado está el acceso "
        "respecto al vecindario de accesos normales."
    )