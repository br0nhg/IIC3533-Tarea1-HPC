import numpy as np

def generate_synthetic_data(N=100000, k=300, seed=42):
    """
    Genera los datos sintéticos X, y, y los coeficientes verdaderos beta_star
    según las instrucciones del ítem (a) de la Tarea 1.
    """
    # Fijamos la semilla para garantizar reproducibilidad
    np.random.seed(seed)
    
    # (I) Muestrear los k+1 coeficientes verdaderos beta* desde N(0,1)
    beta_star = np.random.normal(0, 1, k + 1)
    
    # (II) Generar matriz X de N filas y k columnas desde N(0,1)
    X_features = np.random.normal(0, 1, (N, k))
    
    # Agregar una columna de unos al inicio. 
    # np.ones crea la columna y np.hstack las une horizontalmente.
    unos = np.ones((N, 1))
    X = np.hstack((unos, X_features))
    
    # (III) Calcular y = X * beta* + ruido
    # Generamos N valores de ruido desde N(0,1)
    ruido = np.random.normal(0, 1, N)
    
    # X.dot(beta_star) hace la multiplicación de la matriz por el vector
    y = X.dot(beta_star) + ruido
    
    return X, y, beta_star

# Este bloque solo se ejecuta si corremos este script directamente
if __name__ == "__main__":
    print("Generando datos sintéticos para verificación...")
    X, y, beta_star = generate_synthetic_data()
    
    print("\nResultados de la generación:")
    print(f"Forma de X: {X.shape} (Esperado: 100000, 301)")
    print(f"Forma de y: {y.shape} (Esperado: 100000,)")
    print(f"Forma de beta*: {beta_star.shape} (Esperado: 301,)")
    print("\n¡Etapa 1 completada con éxito!")