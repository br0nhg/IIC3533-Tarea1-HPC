"""
Gráficos de T(p), S(p), E(p) y overhead To(p) — ítems (g) y (h).

S(p) = T(1)/T(p) y E(p) = S(p)/p. La elección de T(1) es parte de lo que el
enunciado pide justificar: acá se usa por defecto el tiempo SECUENCIAL (un for
sin joblib, con BLAS libre), que es la línea base honesta. Con --t1-paralelo se
usa T(1) del backend paralelo, que es lo que se había hecho antes.
"""
import argparse

import matplotlib
matplotlib.use("Agg")  # guardar imágenes sin abrir ventanas (necesario en WSL)
import matplotlib.pyplot as plt
import numpy as np

from config import carpeta_resultados, etiqueta_maquina

VERSIONES = [
    ("auto", "bs_auto", "red", "o-"),
    ("sklearn", "bs_sklearn", "orange", "s-"),
    ("numpy", "bs_numpy", "green", "^-"),
]


def generar_graficos(etiqueta=None, t1_paralelo=False):
    etiqueta = etiqueta or etiqueta_maquina()
    carpeta = carpeta_resultados(etiqueta)

    data = np.load(carpeta / "benchmark_results.npy", allow_pickle=True).item()
    p = np.array(data["p"])
    tiempos = {k: np.array(data[k], dtype=float) for k, *_ in VERSIONES}

    # --- Elección de T(1) (ítem g) ---
    # Nota: para bs_auto no existe una línea base "sin joblib": BaggingRegressor
    # siempre pasa por joblib y n_jobs=1 ES su modo secuencial. Por eso su T(1)
    # se toma del backend paralelo aunque las otras dos usen el tiempo serial.
    base = {}
    for clave, *_ in VERSIONES:
        serial = data.get(f"serial_{clave}")
        if t1_paralelo or serial is None or not np.isfinite(serial):
            base[clave] = tiempos[clave][0]   # T(1) con el backend paralelo
        else:
            base[clave] = float(serial)       # tiempo secuencial real
    origen_t1 = "T(1) paralelo" if t1_paralelo else "secuencial (sin joblib)"

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    sufijo = f"{etiqueta} · $p_{{máx}}={data.get('cores_logicos', p[-1])}$"

    # ----------------- 1. Tiempos T(p) -----------------
    for clave, nombre, color, marca in VERSIONES:
        if np.isfinite(tiempos[clave]).any():
            axs[0].plot(p, tiempos[clave], marca, label=nombre, color=color)
    axs[0].plot(p, base["numpy"] / p, "k--", label="Ideal (NumPy)")
    axs[0].set_title(f"Tiempo de Ejecución $T(p)$ — {sufijo}", fontsize=13)
    axs[0].set_xlabel("Número de procesos ($p$)", fontsize=12)
    axs[0].set_ylabel("Tiempo (segundos)", fontsize=12)
    axs[0].legend()

    # ----------------- 2. Speedup S(p) -----------------
    speedups = {k: base[k] / tiempos[k] for k, *_ in VERSIONES}
    for clave, nombre, color, marca in VERSIONES:
        if np.isfinite(speedups[clave]).any():
            axs[1].plot(p, speedups[clave], marca, label=nombre, color=color)
    axs[1].plot(p, p, "k--", label="Ideal ($S=p$)")
    axs[1].set_title(f"Speedup $S(p)$ — $T(1)$: {origen_t1}", fontsize=13)
    axs[1].set_xlabel("Número de procesos ($p$)", fontsize=12)
    axs[1].set_ylabel("Speedup", fontsize=12)
    axs[1].legend()

    # ----------------- 3. Eficiencia E(p) -----------------
    for clave, nombre, color, marca in VERSIONES:
        e = speedups[clave] / p
        if np.isfinite(e).any():
            axs[2].plot(p, e, marca, label=nombre, color=color)
    axs[2].plot(p, np.ones_like(p), "k--", label="Ideal ($E=1$)")
    axs[2].set_title(f"Eficiencia $E(p)$ — {sufijo}", fontsize=13)
    axs[2].set_xlabel("Número de procesos ($p$)", fontsize=12)
    axs[2].set_ylabel("Eficiencia", fontsize=12)
    axs[2].legend()

    fig.tight_layout()
    fig.savefig(carpeta / "graficos_metricas.png", dpi=300)
    print(f"Gráficos guardados en {carpeta}/graficos_metricas.png")

    # ----------------- 4. Overhead To(p) = p*T(p) - T(1) -----------------
    fig2, ax = plt.subplots(figsize=(7, 5))
    for clave, nombre, color, marca in VERSIONES:
        to = p * tiempos[clave] - base[clave]
        if np.isfinite(to).any():
            ax.plot(p, to, marca, label=nombre, color=color)
    # raw string: '\c' no es una secuencia de escape válida y emite SyntaxWarning
    ax.set_title(r"Overhead $T_o(p) = p \cdot T(p) - T(1)$" + f" — {sufijo}", fontsize=13)
    ax.set_xlabel("Número de procesos ($p$)", fontsize=12)
    ax.set_ylabel("Overhead (segundos·proceso)", fontsize=12)
    ax.legend()
    fig2.tight_layout()
    fig2.savefig(carpeta / "grafico_overhead.png", dpi=300)
    print(f"Gráfico de overhead guardado en {carpeta}/grafico_overhead.png")

    # ----------------- Tabla de resumen para el informe -----------------
    print(f"\n{'p':<4} | " + " | ".join(f"{n:<10} S(p)" for _, n, _, _ in VERSIONES))
    print("-" * 60)
    for i, pi in enumerate(p):
        fila = f"{pi:<4} | "
        for clave, *_ in VERSIONES:
            fila += f"{tiempos[clave][i]:>7.2f}s {speedups[clave][i]:>5.2f} | "
        print(fila)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gráficos de T(p), S(p), E(p) y To(p)")
    parser.add_argument("--maquina", default=None,
                        help="etiqueta de la máquina (default: hostname)")
    parser.add_argument("--t1-paralelo", action="store_true",
                        help="usar T(1) del backend paralelo en vez del tiempo secuencial")
    args = parser.parse_args()
    generar_graficos(etiqueta=args.maquina, t1_paralelo=args.t1_paralelo)
