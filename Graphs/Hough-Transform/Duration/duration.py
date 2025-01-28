import json
import os
import numpy as np
import matplotlib.pyplot as plt

# Lista de concentraciones de fibra a comparar
concentraciones = ["25", "50", "100", "200", "400", "800"]

# Directorio base donde se encuentran los archivos filtrados
base_dir = "Particle-Tracking-Velocimetry\\Hough-Transform"

# Define el tamaño de cada intervalo (en frames)
bin_width = 20

def get_track_lengths(json_file):
    """
    Lee 'json_file', extrae la longitud de cada fibra (número de frames).
    Retorna un array con dichas longitudes.
    """
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    track_lengths = []
    for fiber_id, fiber_data in data.items():
        # Omite llaves especiales
        if fiber_id in ("ruta", "fibras_por_frame"):
            continue

        frames_info = fiber_data.get("frame", [])
        track_length = len(frames_info)
        track_lengths.append(track_length)

    return np.array(track_lengths)

def main():
    plt.figure(figsize=(8, 5))

    for conc in concentraciones:
        # Construimos la ruta al JSON filtrado
        archivo_fibras_filtrado = os.path.join(base_dir, f"fibras_{conc}_filtrado.json")

        # Extraemos las longitudes de trackeo
        track_lengths = get_track_lengths(archivo_fibras_filtrado)
        if track_lengths.size == 0:
            print(f"No se encontraron datos para concentración {conc}. Se omite.")
            continue

        # Calculamos la distribución con bins de 10 en 10
        max_len = track_lengths.max()
        bins = np.arange(0, max_len + bin_width, bin_width)

        counts, bin_edges = np.histogram(track_lengths, bins=bins)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Filtramos intervalos con centro >= 20 (opcional)
        mask_20plus = (bin_centers >= 20)
        x_data = bin_centers[mask_20plus]
        y_data = counts[mask_20plus]

        # Si deseas mostrar todo (incluyendo <20), comenta las dos líneas anteriores
        if len(x_data) < 1:
            print(f"No hay datos suficientes >= 20 frames para {conc}.")
            continue

        # Normalizar dividiendo por el máximo de esta curva
        y_max = y_data.max()
        if y_max == 0:
            print(f"La distribución para {conc} en x>=20 es toda cero.")
            continue

        y_data_norm = y_data / y_max

        # Graficamos la curva normalizada (con marcadores para visualizar mejor)
        plt.plot(x_data, y_data_norm, linestyle='-', label=conc)

    plt.title("Comparación de la distribución de longitud de trackeo (bins de 10 frames)\n(x ≥ 20, normalizada a su valor máximo)")
    plt.xlabel("Longitud del track (frames)")
    plt.ylabel("Conteo normalizado (max = 1)")
    plt.grid(True)
    plt.legend(title="Concentración")
    plt.tight_layout()
    plt.savefig("Graphs/Hough-Transform/Duration/Graphs/track_length_distribution.png")

if __name__ == "__main__":
    main()
