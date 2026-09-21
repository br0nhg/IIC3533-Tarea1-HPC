import time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import BaggingRegressor
from data_utils import generate_synthetic_data

def run_bs_auto(p=2, B=48):
    # 1. Generar los datos
    print(f"Generando datos... (p={p}, B={B})")
    X, y, beta_star = generate_synthetic_data()
    
    # 2. Configurar el modelo base y el ensamblador (Bagging)
    # fit_intercept=False porque X ya incluye la columna de unos
    modelo_base = LinearRegression(fit_intercept=False)
    
    # BaggingRegressor maneja el bootstrap internamente
    # n_estimators es nuestro B (48 resamples)
    # n_jobs es nuestro p (número de procesos)
    modelo_bagging = BaggingRegressor(
        estimator=modelo_base, 
        n_estimators=B, 
        n_jobs=p, 
        bootstrap=True, 
        max_samples=1.0 # Muestrea N elementos con reemplazo
    )
    
    # 3. Iniciar temporizador y ajustar el modelo
    print(f"Ejecutando Bootstrapping con BaggingRegressor (n_jobs={p})...")
    inicio = time.time()
    
    modelo_bagging.fit(X, y)
    
    fin = time.time()
    tiempo_total = fin - inicio
    
    # 4. Calcular el intervalo de confianza (Paso 3 del enunciado)
    # Extraemos los coeficientes de los 48 modelos entrenados
    beta_hats = np.array([estimador.coef_ for estimador in modelo_bagging.estimators_])
    
    # Calculamos los percentiles 2.5 y 97.5 para cada columna (cada coeficiente)
    limite_inferior = np.percentile(beta_hats, 2.5, axis=0)
    limite_superior = np.percentile(beta_hats, 97.5, axis=0)
    
    print(f"\n¡Cálculo completado en {tiempo_total:.4f} segundos!")
    print(f"Forma de la matriz de coeficientes estimados: {beta_hats.shape} (Esperado: 48, 301)")
    
    # Mostramos el intervalo del primer coeficiente (beta_0) como comprobación
    print(f"\nComprobación para beta_0:")
    print(f"Valor verdadero (beta*): {beta_star[0]:.4f}")
    print(f"Intervalo 95%: [{limite_inferior[0]:.4f}, {limite_superior[0]:.4f}]")

if __name__ == "__main__":
    # Probaremos inicialmente con 2 procesos
    run_bs_auto(p=2)