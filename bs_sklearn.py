import time
import numpy as np
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
from data_utils import generate_synthetic_data

def ajustar_bootstrap(X, y, semilla):
    """Esta función es la tarea individual que ejecutará cada proceso."""
    np.random.seed(semilla)
    N = X.shape[0]
    
    # (I) Sortear N índices con reemplazo
    indices = np.random.choice(N, size=N, replace=True)
    
    # (II) Tomar las filas correspondientes para formar X^(b) e y^(b)
    X_b = X[indices]
    y_b = y[indices]
    
    # (III) Calcular beta_hat
    modelo = LinearRegression(fit_intercept=False)
    modelo.fit(X_b, y_b)
    
    return modelo.coef_

def run_bs_sklearn(p=2, B=48):
    print(f"Generando datos... (p={p}, B={B})")
    X, y, beta_star = generate_synthetic_data()
    
    print(f"Ejecutando Bootstrapping con joblib.Parallel (n_jobs={p})...")
    inicio = time.time()
    
    # Paralelismo de Tareas con joblib
    # Pasamos una semilla distinta a cada tarea para que los resamples sean diferentes
    beta_hats_lista = Parallel(n_jobs=p)(
        delayed(ajustar_bootstrap)(X, y, semilla=42+b) for b in range(B)
    )
    
    fin = time.time()
    
    # Convertimos la lista de resultados a una matriz de NumPy
    beta_hats = np.array(beta_hats_lista)
    
    limite_inferior = np.percentile(beta_hats, 2.5, axis=0)
    limite_superior = np.percentile(beta_hats, 97.5, axis=0)
    
    print(f"\n¡Cálculo completado en {fin - inicio:.4f} segundos!")
    print(f"Intervalo 95% beta_0: [{limite_inferior[0]:.4f}, {limite_superior[0]:.4f}] (Verdadero: {beta_star[0]:.4f})")

if __name__ == "__main__":
    run_bs_sklearn(p=2)