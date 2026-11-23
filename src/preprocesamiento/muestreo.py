import numpy as np, pandas as pd
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
CSV = RAIZ / "datos" / "dataset" / "rba-dataset.csv"
SALIDA = RAIZ / "datos" / "muestra.csv"
N_NORMALES = 2000
rng = np.random.default_rng(42)

# Pasada 1: lista de usuarios y de usuarios con toma de cuenta
usuarios, ato = set(), set()
for ch in pd.read_csv(CSV, usecols=["User ID", "Is Account Takeover"],
                      chunksize=1_000_000):
    usuarios.update(ch["User ID"].unique())
    ato.update(ch.loc[ch["Is Account Takeover"] == True, "User ID"].unique())
print("Usuarios totales:", len(usuarios))
print("Usuarios con toma de cuenta:", len(ato))

normales = np.array(sorted(usuarios - ato))
elegidos = set(rng.choice(normales, N_NORMALES, replace=False).tolist()) | ato

# Pasada 2: guardar TODOS los logins de los usuarios elegidos
primera = True
for ch in pd.read_csv(CSV, chunksize=1_000_000):
    parte = ch[ch["User ID"].isin(elegidos)]
    parte.to_csv(SALIDA, mode="w" if primera else "a", header=primera, index=False)
    primera = False
print("Listo:", SALIDA)