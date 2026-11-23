import numpy as np, joblib
import streamlit as st
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
MODELOS = RAIZ / "modelos"
PROC = RAIZ / "datos" / "procesado"

st.set_page_config(page_title="Detector de logins sospechosos", page_icon="🔐")
st.title("🔐 Detector de inicios de sesión sospechosos")
st.caption("Modelo: Local Outlier Factor (LOF), no supervisado")

# ---- Comprobar que existen el modelo y el escalador ----
if not (MODELOS / "scaler.joblib").exists():
    st.error("No encuentro el modelo. Ejecuta primero preparar.py y entrenar_lof.py.")
    st.stop()

ks = sorted(int(p.stem.split("_k")[1]) for p in MODELOS.glob("lof_k*.joblib"))
k = st.sidebar.selectbox("Número de vecinos (k)", ks, index=ks.index(20) if 20 in ks else 0)
pct = st.sidebar.selectbox("Marcar como sospechoso el % más alto", [0.5, 1, 2, 5], index=1)

scaler = joblib.load(MODELOS / "scaler.joblib")
lof = joblib.load(MODELOS / f"lof_k{k}.joblib")

# Umbral: percentil de los puntajes de prueba que dejó el entrenamiento
umbral = np.percentile(np.load(PROC / f"scores_k{k}.npy"), 100 - pct)

# ---- Casos de prueba para la demostración ----
CASOS = {
    "Login normal": dict(hora=14, ok=1, pais=0, asn=0, ip=0, disp=0, horas=24.0),
    "3 a. m. desde un país nuevo": dict(hora=3, ok=1, pais=1, asn=1, ip=1, disp=0, horas=300.0),
    "Todo nuevo y fallido": dict(hora=4, ok=0, pais=1, asn=1, ip=1, disp=1, horas=2.0),
    "Personalizado": dict(hora=12, ok=1, pais=0, asn=0, ip=0, disp=0, horas=24.0),
}
caso = st.selectbox("Caso de prueba", list(CASOS))
c = CASOS[caso]

col1, col2 = st.columns(2)
with col1:
    hora = st.slider("Hora del acceso", 0, 23, c["hora"], key=f"{caso}_hora")
    horas = st.number_input("Horas desde el último login", 0.0, 10000.0,
                            c["horas"], key=f"{caso}_horas")
    ok = st.checkbox("Login exitoso", bool(c["ok"]), key=f"{caso}_ok")
with col2:
    pais = st.checkbox("País nuevo para este usuario", bool(c["pais"]), key=f"{caso}_pais")
    asn = st.checkbox("Red (ASN) nueva", bool(c["asn"]), key=f"{caso}_asn")
    ip = st.checkbox("IP nueva", bool(c["ip"]), key=f"{caso}_ip")
    disp = st.checkbox("Dispositivo nuevo", bool(c["disp"]), key=f"{caso}_disp")

if st.button("Evaluar login", type="primary"):
    # El orden debe ser el mismo que FEATS en variables.py
    x = np.array([[np.sin(2 * np.pi * hora / 24), np.cos(2 * np.pi * hora / 24),
                   int(ok), int(pais), int(asn), int(ip), int(disp),
                   np.log1p(horas * 3600)]])
    puntaje = float(-lof.score_samples(scaler.transform(x))[0])

    if puntaje > umbral:
        st.error("⚠️ SOSPECHOSO: conviene revisar este acceso")
    else:
        st.success("✅ NORMAL")
    st.metric("Puntaje LOF", f"{puntaje:.2f}", f"umbral {umbral:.2f}", delta_color="off")
    st.caption("Cerca de 1 es normal. Mientras más alto, más aislado está el acceso "
               "de los accesos parecidos.")