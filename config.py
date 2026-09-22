"""
Utilidades compartidas por los scripts de experimentos.

El objetivo es que correr los benchmarks en la segunda máquina no sobrescriba
los resultados de la primera: todas las salidas se guardan bajo
resultados/<etiqueta>/, donde <etiqueta> es por defecto el hostname.
"""
import os
import socket
from pathlib import Path


def etiqueta_maquina():
    """Nombre corto que identifica esta máquina. Se puede forzar con MAQUINA=..."""
    return os.environ.get("MAQUINA") or socket.gethostname().split(".")[0]


def carpeta_resultados(etiqueta=None):
    """Crea (si no existe) y devuelve resultados/<etiqueta>/."""
    carpeta = Path("resultados") / (etiqueta or etiqueta_maquina())
    carpeta.mkdir(parents=True, exist_ok=True)
    return carpeta


def cores_logicos():
    """p_máx = número de cores lógicos. os.cpu_count() los cuenta con hyperthreading."""
    return os.cpu_count() or 1


def valores_p(p_max=None):
    """Lista de p a evaluar: todos los enteros de 1 a p_máx (ítem f)."""
    return list(range(1, (p_max or cores_logicos()) + 1))


def valores_grilla(p_max=None):
    """
    Valores de p y t para la grilla del ítem (i).

    Con p_máx grande, recorrer todos los enteros sería carísimo, así que se usan
    potencias de 2 más el propio p_máx. La restricción p*t <= p_máx se aplica
    después, al recorrer la grilla.
    """
    p_max = p_max or cores_logicos()
    vals = [v for v in (1, 2, 4, 8, 16, 32, 64) if v <= p_max]
    if p_max not in vals:
        vals.append(p_max)
    return sorted(vals)


def describir_entorno():
    """Texto con el entorno de ejecución, para dejarlo registrado en los logs."""
    import joblib
    import numpy as np
    import threadpoolctl

    lineas = [
        f"Máquina:        {etiqueta_maquina()}",
        f"Cores lógicos:  {cores_logicos()}",
        f"NumPy:          {np.__version__}",
        f"joblib:         {joblib.__version__}",
        f"threadpoolctl:  {threadpoolctl.__version__}",
    ]
    # Forzamos la carga de BLAS: threadpool_info() devuelve [] si todavía no se usó.
    np.linalg.solve(np.eye(4), np.ones(4))
    for lib in threadpoolctl.threadpool_info():
        if lib["user_api"] == "blas":
            lineas.append(
                f"BLAS:           {lib['internal_api']} {lib.get('version', '?')} "
                f"({lib['num_threads']} threads por defecto)"
            )
    return "\n".join(lineas)


if __name__ == "__main__":
    print(describir_entorno())
    print(f"\nSalidas irán a: {carpeta_resultados()}")
