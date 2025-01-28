import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Lista de concentraciones a procesar
concentraciones = ["25", "50", "100", "200", "400", "800"]

# Directorio base (ajusta según tu estructura)
base_dir = "Particle-Tracking-Velocimetry\\Hough-Transform"

def plot_trajectories_for_concentration(concentracion):
    """
    Carga el archivo JSON filtrado correspondiente a 'concentracion'
    y genera un gráfico con todas sus trayectorias, guardándolo en un .png.
    """
    # Construimos la ruta del archivo para la concentración dada
    archivo_fibras_filtrado = os.path.join(base_dir, f"fibras_{concentracion}_filtrado.json")

    # Carga de datos
    with open(archivo_fibras_filtrado, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Configuramos la figura
    plt.figure(figsize=(8, 8))
    plt.xlim(0, 1024)
    plt.ylim(1024, 0)  # Invertimos el eje Y

    # Recorremos todas las fibras (excepto metadatos) y graficamos
    for fiber_id, fiber_data in data.items():
        if fiber_id in ("ruta", "fibras_por_frame"):
            continue

        centroids = fiber_data.get("centroide", [])
        if not centroids:
            continue

        centroids = np.array(centroids)
        x_coords = centroids[:, 0]
        y_coords = centroids[:, 1]

        # Dibuja la trayectoria como una línea
        plt.plot(x_coords, y_coords, linewidth=1, alpha=0.6)

    # Etiquetas y títulos
    plt.title(f"Trayectorias Trackeadas (fibras_{concentracion})")
    plt.xlabel("X (píxeles)")
    plt.ylabel("Y (píxeles)")
    plt.grid(True)

    plt.tight_layout()
    
    # Guardamos la figura con un nombre que incluya la concentración
    nombre_salida = f"Graphs/Hough-Transform/Trayectories/Graphs/trayectorias_{concentracion}.png"
    plt.savefig(nombre_salida, dpi=300)
    plt.close()  # Cerramos la figura para liberar memoria

def main():
    # Iteramos sobre la lista de concentraciones y generamos un gráfico por cada una
    for conc in concentraciones:
        print(f"Generando gráfico de trayectorias para concentración: {conc}")
        plot_trajectories_for_concentration(conc)

if __name__ == "__main__":
    main()
