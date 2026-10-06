import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Fijamos una semilla para que los resultados aleatorios sean reproducibles
np.random.seed(42)

# Generación de 100 puntos aleatorios en el plano [-2, 2] x [-2, 2]
n_puntos = 100
# Generamos un arreglo de forma (100, 2)
X_train = np.random.uniform(low=-2.0, high=2.0, size=(n_puntos, 2))

# Separamos las coordenadas X e Y de los puntos para facilitar los cálculos
x_coords = X_train[:, 0]
y_coords = X_train[:, 1]

# Cálculo de los valores verdaderos usando la función no lineal sin(x)*cos(y)
# Usaremos 'z' para representar el valor de la función
z_verdadero = np.sin(x_coords) * np.cos(y_coords)

# Adición de un pequeño ruido gaussiano
# Utilizamos una media de 0 y una desviación estándar pequeña (ej. 0.1)
ruido = np.random.normal(loc=0.0, scale=0.1, size=n_puntos)
z_ruidoso = z_verdadero + ruido

# --- Visualización de los datos generados ---
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Graficamos los puntos dispersos (scatter) con su respectivo ruido
scatter = ax.scatter(x_coords, y_coords, z_ruidoso, 
                     c=z_ruidoso, cmap='viridis', 
                     marker='o', label='Datos de entrenamiento (con ruido)')

ax.set_title('Datos Sintéticos Generados')
ax.set_xlabel('Eje X')
ax.set_ylabel('Eje Y')
ax.set_zlabel('Valor Z')
fig.colorbar(scatter, label='Magnitud de Z')
plt.legend()
plt.show()

def rbf_kernel(r, epsilon):
    """
    Evalúa el núcleo gaussiano para una matriz de distancias r.
    """
    return np.exp(-(epsilon * r)**2)

def construir_matriz_phi(X, centros, epsilon):
    """
    Construye la matriz de diseño Phi calculando las distancias euclídeas
    y aplicando la función de base radial.
    """
    # Calculamos la matriz de distancias euclídeas ||x - x_j||
    # La función cdist devuelve una matriz donde el elemento (i, j) 
    # es la distancia entre X[i] y centros[j]
    distancias = cdist(X, centros, metric='euclidean')
    
    # Aplicamos el parámetro de forma epsilon y el kernel gaussiano
    Phi = rbf_kernel(distancias, epsilon)
    
    return Phi

# Los centros de las funciones gaussianas son los mismos puntos de entrenamiento
centros = X_train

# Definimos un parámetro de forma epsilon inicial para probar
# Un valor de 1.0 suele ser un punto de partida neutral
epsilon_prueba = 1.0 

# Construimos la matriz de diseño para nuestros datos de entrenamiento
Phi_train = construir_matriz_phi(X_train, centros, epsilon_prueba)

# Verificamos las dimensiones resultantes
print(f"Dimensiones de la matriz de distancias / matriz Phi: {Phi_train.shape}")
print(f"Muestra de los primeros 3x3 elementos de Phi:\n{Phi_train[:3, :3]}")

# ==========================================
# Entrenamiento y Regularización
# ==========================================

def entrenar_rbf(Phi, y, lambda_reg=1e-6):
    """
    Resuelve el sistema lineal regularizado (Phi + lambda * I)w = y
    para encontrar los pesos w del modelo RBF.
    """
    # Obtenemos la dimensión de la matriz cuadrada Phi
    n = Phi.shape[0]
    
    # Construimos la matriz Identidad (I) del mismo tamaño
    I = np.identity(n)
    
    # Construimos la matriz del lado izquierdo del sistema
    A = Phi + lambda_reg * I
    
    # Resolvemos el sistema lineal A * w = y
    w = np.linalg.solve(A, y)
    
    return w

# --- Prueba de la Fase 3 ---
# Definimos el valor de lambda (un valor pequeño como sugiere el taller)
lambda_prueba = 1e-6

# Calculamos los pesos w usando la matriz Phi_train y los valores z_ruidoso
pesos_w = entrenar_rbf(Phi_train, z_ruidoso, lambda_prueba)

# Verificamos los resultados
print(f"Dimensiones del vector de pesos w: {pesos_w.shape}")
print(f"Muestra de los primeros 5 pesos calculados:\n{pesos_w[:5]}")

# ==========================================
# Evaluación y Visualización
# ==========================================

# Creación de la malla fina (50x50)
x_line = np.linspace(-2.0, 2.0, 50)
y_line = np.linspace(-2.0, 2.0, 50)
X_mesh, Y_mesh = np.meshgrid(x_line, y_line)

# Aplanamos las matrices de la malla para tener una lista de 2500 puntos (x, y)
# np.ravel() convierte la matriz 2D en un arreglo 1D
puntos_malla = np.column_stack((X_mesh.ravel(), Y_mesh.ravel()))

# Evaluación del modelo en la malla
# Calculamos la matriz Phi para los puntos de prueba contra los centros de entrenamiento
Phi_test = construir_matriz_phi(puntos_malla, centros, epsilon_prueba)

# Calculamos las predicciones multiplicando la matriz Phi_test por los pesos w
# Usamos el operador @ para la multiplicación de matrices en numpy (producto punto)
z_pred_aplanado = Phi_test @ pesos_w

# Reconstruimos la forma (50, 50) para poder graficar la superficie
Z_pred = z_pred_aplanado.reshape(X_mesh.shape)

# 3. Visualización de la Superficie
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Graficamos la superficie de predicción con un mapa de colores
superficie = ax.plot_surface(X_mesh, Y_mesh, Z_pred, 
                             cmap='viridis', alpha=0.8, 
                             edgecolor='none', label='Superficie RBF')

# Superponemos los puntos de entrenamiento originales (con ruido)
ax.scatter(x_coords, y_coords, z_ruidoso, 
           color='red', marker='o', s=20, label='Datos de entrenamiento')

ax.set_title(f'Aproximación RBF (epsilon={epsilon_prueba}, lambda={lambda_prueba})')
ax.set_xlabel('Eje X')
ax.set_ylabel('Eje Y')
ax.set_zlabel('Valor Z')

# Añadimos una barra de color para la superficie
fig.colorbar(superficie, ax=ax, shrink=0.5, aspect=10, label='Predicción Z')

# matplotlib 3D no soporta bien la leyenda automática para plot_surface
# pero los scatter points sí aparecerán indicados si forzamos la leyenda
import matplotlib.patches as mpatches
red_patch = mpatches.Patch(color='red', label='Datos de entrenamiento')
surface_patch = mpatches.Patch(color='viridis', label='Superficie Predicha')
plt.legend(handles=[red_patch])

plt.show()

# ==========================================
# Estudio del Parámetro de Forma (Epsilon)
# ==========================================

# Definimos los tres valores de epsilon a estudiar
epsilons = [0.1, 1.0, 10.0] # Pequeño, Óptimo, Grande
titulos = ['Epsilon Pequeño (0.1)', 'Epsilon Óptimo (1.0)', 'Epsilon Grande (10.0)']

fig = plt.figure(figsize=(18, 6))

for i, eps in enumerate(epsilons):
    # 1. Construir matriz de entrenamiento
    Phi_train_eps = construir_matriz_phi(X_train, centros, eps)
    
    # 2. Entrenar el modelo (usamos el mismo lambda pequeño)
    w_eps = entrenar_rbf(Phi_train_eps, z_ruidoso, lambda_reg=1e-6)
    
    # 3. Construir matriz de prueba en la malla
    Phi_test_eps = construir_matriz_phi(puntos_malla, centros, eps)
    
    # 4. Predecir
    z_pred_aplanado_eps = Phi_test_eps @ w_eps
    Z_pred_eps = z_pred_aplanado_eps.reshape(X_mesh.shape)
    
    # 5. Graficar en un subplot
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    ax.plot_surface(X_mesh, Y_mesh, Z_pred_eps, 
                    cmap='viridis', alpha=0.8, edgecolor='none')
    ax.scatter(x_coords, y_coords, z_ruidoso, 
               color='red', marker='o', s=10)
    
    ax.set_title(titulos[i])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_zlim(-2, 2) # Fijamos el límite Z para comparar justamente

plt.tight_layout()
plt.show()