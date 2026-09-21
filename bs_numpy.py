import time
import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_info
from threadpoolctl import threadpool_limits
import pprint
from data_utils import generate_synthetic_data

def ajustar_bootstrap_numpy(X, y, semilla):
    """Resuelve la regresión usando la Ecuación Normal pura con NumPy."""
    np.random.seed(semilla)
    N = X.shape[0]
    
    indices = np.random.choice(N, size=N, replace=True)
    X_b = X[indices]
    y_b = y[indices]
    
    X_T = X_b.T
    X_T_X = X_T.dot(X_b)
    X_T_y = X_T.dot(y_b)
    beta_hat = np.linalg.solve(X_T_X, X_T_y)
    
    return beta_hat

def run_bs_numpy(p=8, B=48):
    # --- NUEVO: Inspeccionando los threads internos ---
    print("=== Información de Threadpool (NumPy) ===")
    info = threadpool_info()
    # Filtramos para mostrar solo la librería matemática principal (BLAS/OpenBLAS/MKL)
    for lib in info:
        if lib['internal_api'] in ('openblas', 'mkl', 'blas'):
            print(f"Librería: {lib['filepath'].split('/')[-1]}")
            print(f"Threads internos por defecto: {lib['num_threads']}")
    print("=========================================\n")
    # ---------------------------------------------------

    print(f"Generando datos... (p={p}, B={B})")
    X, y, beta_star = generate_synthetic_data()
    
    print(f"Ejecutando Bootstrapping con NumPy (n_jobs={p}) y limits=1...")
    inicio = time.time()
    
    # Envolvemos el llamado a Parallel con el limitador de hilos
    with threadpool_limits(limits=1, user_api='blas'):
        beta_hats_lista = Parallel(n_jobs=p)(
            delayed(ajustar_bootstrap_numpy)(X, y, semilla=42+b) for b in range(B)
        )
        
    fin = time.time()
    
    print(f"\n¡Cálculo completado en {fin - inicio:.4f} segundos!")

if __name__ == "__main__":
    # Corremos con p=8 para usar todos tus núcleos lógicos
    run_bs_numpy(p=8)