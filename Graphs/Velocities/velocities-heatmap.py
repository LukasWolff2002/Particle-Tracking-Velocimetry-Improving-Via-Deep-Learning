import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Lista de concentraciones sobre las que iterar
concentracion_fibras_list = ["25", "50", "100", "200", "400", "800"]

# Definimos un tamaño de grilla para la generación de los mapas de calor
GRID_SIZE = (200, 200)  # Ajustar a conveniencia
# Extensión de la imagen (asumiendo que las coordenadas van de 0 a 1024)
# para que en el plot se muestre de [0, 1024] en X y [0, 1024] en Y.
EXTENT = [0, 1024, 0, 1024]

def plot_linear_velocity(concentracion_fibras):
    """
    Carga el JSON de velocidad lineal (vx, vy) para la concentración dada,
    calcula y grafica los mapas promedio de vx y vy en una grilla reducida,
    y guarda los plots resultantes.
    """
    # ----------------------------------------------------------------------------
    # 1) Carga de datos
    # ----------------------------------------------------------------------------
    # Ajusta la ruta si tu archivo se llama de otra forma o está en otra carpeta
    json_file_linear = f"Graphs/Velocities/fibers_{concentracion_fibras}_convolutionated.json"
    
    with open(json_file_linear, "r", encoding="utf-8") as f:
        data_linear = json.load(f)

    # ----------------------------------------------------------------------------
    # 2) Extracción de datos para promedio
    # ----------------------------------------------------------------------------
    centroid_x, centroid_y = [], []
    vel_x, vel_y = [], []

    # Recorre todas las fibras y extrae centroides y velocidades convolucionadas
    for fiber_id, fiber_data in data_linear.items():
        if fiber_id in ["ruta", "fibras_por_frame"]:
            continue
        
        centroids = np.array(fiber_data["centroide"])  # Cada elemento = [x, y]
        if centroids.size == 0:
            continue
        
        x_coords = centroids[:, 0]
        y_coords = centroids[:, 1]

        vx_smooth = fiber_data.get("velocidad_x_convolucionada", [])
        vy_smooth = fiber_data.get("velocidad_y_convolucionada", [])

        # Coincidimos longitudes: solo tomamos hasta donde haya velocidad
        n_vel = min(len(vx_smooth), len(vy_smooth))
        centroid_x.extend(x_coords[:n_vel])
        centroid_y.extend(y_coords[:n_vel])
        vel_x.extend(vx_smooth[:n_vel])
        vel_y.extend(vy_smooth[:n_vel])

    # Convertimos a arrays de NumPy
    centroid_x = np.array(centroid_x)
    centroid_y = np.array(centroid_y)
    vel_x = np.array(vel_x)
    vel_y = np.array(vel_y)

    # ----------------------------------------------------------------------------
    # 3) Cálculo de velocidades promedio en celdas de la grilla
    # ----------------------------------------------------------------------------
    # Histograma 2D para la suma de velocidades
    sum_vel_x, x_edges, y_edges = np.histogram2d(
        centroid_x, centroid_y, bins=GRID_SIZE, weights=vel_x
    )
    sum_vel_y, _, _ = np.histogram2d(
        centroid_x, centroid_y, bins=GRID_SIZE, weights=vel_y
    )
    # Contamos cuántos valores hay en cada celda
    counts, _, _ = np.histogram2d(centroid_x, centroid_y, bins=GRID_SIZE)

    # Evitamos division por cero
    avg_vel_x = np.divide(
        sum_vel_x, counts, out=np.zeros_like(sum_vel_x), where=(counts > 0)
    )
    avg_vel_y = np.divide(
        sum_vel_y, counts, out=np.zeros_like(sum_vel_y), where=(counts > 0)
    )

    # Enmascaramos celdas donde no hay datos
    avg_vel_x = np.ma.masked_where(counts == 0, avg_vel_x)
    avg_vel_y = np.ma.masked_where(counts == 0, avg_vel_y)

    # ----------------------------------------------------------------------------
    # 4) Gráficas para vx y vy
    # ----------------------------------------------------------------------------
    # Normalizaciones centradas en 0, de -max a +max
    norm_x = TwoSlopeNorm(vmin=np.min(avg_vel_x), vcenter=0, vmax=np.max(avg_vel_x))
    norm_y = TwoSlopeNorm(vmin=np.min(avg_vel_y), vcenter=0, vmax=np.max(avg_vel_y))

    # Plot de la velocidad en X
    plt.figure(figsize=(10, 8))
    plt.imshow(
        avg_vel_x.T,
        origin="upper",
        extent=EXTENT,
        aspect="auto",
        cmap="coolwarm",
        norm=norm_x
    )
    plt.colorbar(label="Average Velocity in X (units/s)")
    plt.title(f"Average Velocity in X (Reduced Grid) - Fibras {concentracion_fibras}")
    plt.xlabel("Centroid X Position")
    plt.ylabel("Centroid Y Position")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"Graphs/Velocities/Graphs/average_velocity_x_{concentracion_fibras}.png")

    # Plot de la velocidad en Y
    plt.figure(figsize=(10, 8))
    plt.imshow(
        avg_vel_y.T,
        origin="upper",
        extent=EXTENT,
        aspect="auto",
        cmap="coolwarm",
        norm=norm_y
    )
    plt.colorbar(label="Average Velocity in Y (units/s)")
    plt.title(f"Average Velocity in Y (Reduced Grid) - Fibras {concentracion_fibras}")
    plt.xlabel("Centroid X Position")
    plt.ylabel("Centroid Y Position")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"Graphs/Velocities/Graphs/average_velocity_y_{concentracion_fibras}.png")
 

def plot_angular_velocity(concentracion_fibras):
    """
    Carga el JSON de velocidad angular para la concentración dada,
    calcula y grafica el mapa promedio de velocidad angular en una grilla reducida,
    y guarda el plot resultante.
    """
    # ----------------------------------------------------------------------------
    # 1) Carga de datos
    # ----------------------------------------------------------------------------
    # Ajusta la ruta si tu archivo se llama de otra forma o está en otra carpeta
    json_file_angular = f"Graphs/Velocities/fibers_{concentracion_fibras}_convolutionated.json"

    with open(json_file_angular, "r", encoding="utf-8") as f:
        data_angular = json.load(f)

    # ----------------------------------------------------------------------------
    # 2) Extracción de datos para promedio angular
    # ----------------------------------------------------------------------------
    centroid_x, centroid_y = [], []
    angular_vel_list = []

    for fiber_id, fiber_data in data_angular.items():
        if fiber_id in ["ruta", "fibras_por_frame"]:
            continue
        
        centroids = np.array(fiber_data["centroide"])  # cada elemento = [x, y]
        if centroids.size == 0:
            continue
        
        x_coords = centroids[:, 0]
        y_coords = centroids[:, 1]

        angular_vel_smooth = fiber_data.get("velocidad_angular_convolucionada", [])
        n_ang = len(angular_vel_smooth)

        # Coincidimos longitudes hasta donde haya velocidad angular
        centroid_x.extend(x_coords[:n_ang])
        centroid_y.extend(y_coords[:n_ang])
        angular_vel_list.extend(angular_vel_smooth[:n_ang])

    # Convertimos a arrays
    centroid_x = np.array(centroid_x)
    centroid_y = np.array(centroid_y)
    angular_vel_list = np.array(angular_vel_list)

    # ----------------------------------------------------------------------------
    # 3) Cálculo de velocidades angulares promedio en celdas
    # ----------------------------------------------------------------------------
    sum_angular_vel, x_edges, y_edges = np.histogram2d(
        centroid_x, centroid_y, bins=GRID_SIZE, weights=angular_vel_list
    )
    counts, _, _ = np.histogram2d(centroid_x, centroid_y, bins=GRID_SIZE)

    avg_angular_vel = np.divide(
        sum_angular_vel, counts, out=np.zeros_like(sum_angular_vel), where=(counts > 0)
    )

    # Enmascaramos celdas sin datos
    avg_angular_vel = np.ma.masked_where(counts == 0, avg_angular_vel)

    # ----------------------------------------------------------------------------
    # 4) Gráfica de velocidad angular
    # ----------------------------------------------------------------------------
    # Normalización centrada en 0
    norm = TwoSlopeNorm(
        vmin=np.min(avg_angular_vel),
        vcenter=0,
        vmax=np.max(avg_angular_vel)
    )

    plt.figure(figsize=(10, 8))
    plt.imshow(
        avg_angular_vel.T,
        origin="upper",
        extent=EXTENT,
        aspect="auto",
        cmap="coolwarm",
        norm=norm
    )
    plt.colorbar(label="Average Angular Velocity (degrees/s)")
    plt.title(f"Average Angular Velocity (Reduced Grid) - Fibras {concentracion_fibras}")
    plt.xlabel("Centroid X Position")
    plt.ylabel("Centroid Y Position")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"Graphs/Velocities/Graphs/average_angular_velocity_{concentracion_fibras}.png")
  

# =============================================================================
# Bucle principal: para cada concentración, graficar velocidades lineales y angulares
# =============================================================================
if __name__ == "__main__":
    for fibras in concentracion_fibras_list:
        print(f"\n=== Procesando concentración de fibras: {fibras} ===\n")
        
        # 1) Graficar velocidades lineales (x e y)
        plot_linear_velocity(fibras)
        
        # 2) Graficar velocidad angular
        plot_angular_velocity(fibras)
