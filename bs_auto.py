import argparse
import time

import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression

from config import cores_logicos
from data_utils import generate_synthetic_data


def run_bs_auto(p=None, B=48, semilla=42, N=100000, k=300):
    p = p or cores_logicos()

    # 1. Generar los datos
    print(f"Generando datos... (p={p}, B={B})")
    X, y, beta_star = generate_synthetic_data(N=N, k=k)

    # 2. Configurar el modelo base y el ensamblador (Bagging)
    # fit_intercept=False porque X ya incluye la columna de unos
    modelo_base = LinearRegression(fit_intercept=False)

    # BaggingRegressor maneja el bootstrap internamente
    # n_estimators es nuestro B (48 resamples)
    # n_jobs es nuestro p (número de procesos)
    # random_state fija el RNG interno: sin esto los resamples cambian en cada
    # ejecución y los resultados NO son reproducibles (ítem c).
    modelo_bagging = BaggingRegressor(
        estimator=modelo_base,
        n_estimators=B,
        n_jobs=p,
        bootstrap=True,
        max_samples=1.0,  # Muestrea N elementos con reemplazo
        random_state=semilla,
    )

    # 3. Iniciar temporizador y ajustar el modelo
    print(f"Ejecutando Bootstrapping con BaggingRegressor (n_jobs={p})...")
    inicio = time.time()

    modelo_bagging.fit(X, y)

    tiempo_total = time.time() - inicio

    # 4. Calcular el intervalo de confianza (Paso 3 del enunciado)
    # Extraemos los coeficientes de los B modelos entrenados
    beta_hats = np.array([est.coef_ for est in modelo_bagging.estimators_])

    # Percentiles 2.5 y 97.5 para cada columna (cada coeficiente)
    limite_inferior = np.percentile(beta_hats, 2.5, axis=0)
    limite_superior = np.percentile(beta_hats, 97.5, axis=0)

    print(f"\n¡Cálculo completado en {tiempo_total:.4f} segundos!")
    print(f"Forma de la matriz de coeficientes estimados: {beta_hats.shape} "
          f"(Esperado: {B}, {X.shape[1]})")

    cobertura = np.mean((limite_inferior <= beta_star) & (beta_star <= limite_superior))
    print(f"\nComprobación para beta_0:")
    print(f"Valor verdadero (beta*): {beta_star[0]:.4f}")
    print(f"Intervalo 95%: [{limite_inferior[0]:.4f}, {limite_superior[0]:.4f}]")
    print(f"Cobertura de beta* por los intervalos: {cobertura:.1%}")

    return tiempo_total, beta_hats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrapping OLS con BaggingRegressor")
    parser.add_argument("-p", type=int, default=None,
                        help="número de procesos (default: cores lógicos)")
    parser.add_argument("-B", type=int, default=48, help="número de resamples")
    parser.add_argument("-N", type=int, default=100000, help="observaciones")
    parser.add_argument("-k", type=int, default=300, help="variables de entrada")
    args = parser.parse_args()
    run_bs_auto(p=args.p, B=args.B, N=args.N, k=args.k)
