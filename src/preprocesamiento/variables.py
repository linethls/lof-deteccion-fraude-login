import numpy as np, pandas as pd
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
SAL = RAIZ / "datos" / "procesado"
SAL.mkdir(parents=True, exist_ok=True)

cols = ["Login Timestamp", "User ID", "IP Address", "Country", "ASN",
        "Browser Name and Version", "OS Name and Version", "Device Type",
        "Login Successful", "Is Account Takeover"]
# keep_default_na=False evita que el país "NA" (Namibia) se lea como vacío
df = pd.read_csv(RAIZ / "datos" / "muestra.csv", usecols=cols,
                 dtype=str, keep_default_na=False)

df["Login Timestamp"] = pd.to_datetime(df["Login Timestamp"])
df = df.sort_values(["User ID", "Login Timestamp"]).reset_index(drop=True)

df["dispositivo"] = (df["Device Type"] + "|" + df["OS Name and Version"]
                     + "|" + df["Browser Name and Version"])
df["login_ok"] = (df["Login Successful"] == "True").astype(int)
y = (df["Is Account Takeover"] == "True").astype(int)

# Hora cíclica
h = df["Login Timestamp"].dt.hour + df["Login Timestamp"].dt.minute / 60
df["hora_sin"] = np.sin(2 * np.pi * h / 24)
df["hora_cos"] = np.cos(2 * np.pi * h / 24)

# Novedad por usuario (1 = nunca visto antes; el primer login cuenta como 0)
g = df.groupby("User ID")
primero = g.cumcount() == 0
for col, nombre in [("Country", "pais_nuevo"), ("ASN", "asn_nuevo"),
                    ("IP Address", "ip_nueva"), ("dispositivo", "dispositivo_nuevo")]:
    nuevo = df.groupby(["User ID", col]).cumcount() == 0
    df[nombre] = (nuevo & ~primero).astype(int)

# Tiempo desde el último login (segundos, en logaritmo)
seg = g["Login Timestamp"].diff().dt.total_seconds()
df["seg_desde_ultimo"] = np.log1p(seg.fillna(seg.median()))

FEATS = ["hora_sin", "hora_cos", "login_ok", "pais_nuevo", "asn_nuevo",
         "ip_nueva", "dispositivo_nuevo", "seg_desde_ultimo"]
salida = df[FEATS].copy()
salida["y"] = y
salida["usuario"] = df["User ID"]
salida.to_csv(SAL / "features.csv", index=False)
print("Filas:", len(salida), "| tomas de cuenta:", y.sum())
print(salida[FEATS].describe().T[["mean", "min", "max"]].round(3))