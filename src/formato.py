"""Funciones para que la salida de la consola se vea ordenada y se pueda capturar."""
import textwrap
import numpy as np

ANCHO = 78


def miles(n):
    """12345 -> '12.345' (estilo español)."""
    return f"{int(n):,}".replace(",", ".")


def titulo(paso, nombre, descripcion):
    """Cabecera del paso: número, nombre y qué hace."""
    print("=" * ANCHO)
    print(f" PASO {paso}: {nombre}")
    print("=" * ANCHO)
    print(" Qué hace:")
    for linea in textwrap.wrap(descripcion, ANCHO - 4):
        print("   " + linea)
    print("-" * ANCHO)


def seccion(texto):
    print(f"\n>> {texto}")


def nota(puntos, etiqueta="Cómo leer este resultado"):
    """Cierre del paso: lista de puntos que explican la salida."""
    print("\n" + "-" * ANCHO)
    print(f" {etiqueta}:")
    for p in puntos:
        for i, linea in enumerate(textwrap.wrap(p, ANCHO - 6)):
            print(("  - " if i == 0 else "    ") + linea)
    print("=" * ANCHO)


def tabla(df, decimales=3, izquierda=None):
    """Imprime un DataFrame como tabla con bordes.

    decimales: entero, o diccionario {columna: decimales}.
    izquierda: columnas alineadas a la izquierda (por defecto, la primera).
    """
    cols = list(df.columns)
    izquierda = set(izquierda) if izquierda else {cols[0]}

    def fmt(v, col):
        if isinstance(v, (float, np.floating)):
            d = decimales.get(col, 3) if isinstance(decimales, dict) else decimales
            return f"{v:.{d}f}"
        return str(v)

    filas = [[fmt(r[c], c) for c in cols] for _, r in df.iterrows()]
    anchos = [max([len(str(c))] + [len(f[i]) for f in filas])
              for i, c in enumerate(cols)]
    sep = "+" + "+".join("-" * (a + 2) for a in anchos) + "+"

    def linea(vals):
        celdas = []
        for i, v in enumerate(vals):
            alin = v.ljust(anchos[i]) if cols[i] in izquierda else v.rjust(anchos[i])
            celdas.append(f" {alin} ")
        return "|" + "|".join(celdas) + "|"

    print(sep)
    print(linea([str(c) for c in cols]))
    print(sep)
    for f in filas:
        print(linea(f))
    print(sep)