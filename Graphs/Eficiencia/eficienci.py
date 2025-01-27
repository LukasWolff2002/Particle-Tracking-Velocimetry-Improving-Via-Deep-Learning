import json
import matplotlib.pyplot as plt

# Cambia esta variable si deseas usarla en otra parte del código


numero_fibras = "800"

archivo_fibras_filtrado = f"Particle-Tracking-Velocimetry\\YOLO\\fibras_{numero_fibras}_filtrado.json"


def compute_fibers_tracked_by_frame(datos):
    """
    Dado un diccionario (contenido de un JSON filtrado),
    retorna (frames_eje_x, fibras_acumuladas_eje_y), donde:
      - frames_eje_x = [1, 2, ..., max_frame_en_datos]
      - fibras_acumuladas_eje_y[i] = cuántas fibras han aparecido
        en frame <= frames_eje_x[i].
    """
    fibra_min_frame = {}
    max_frame_global = 0

    for key, value in datos.items():
        # Saltamos claves especiales
        if key in ("ruta", "fibras_por_frame"):
            continue

        lista_frames = value.get("frame", [])
        if not lista_frames:
            continue

        # frames es algo como [[1],[2],...], lo convertimos a [1,2,...]
        frames_planos = [f[0] for f in lista_frames]
        min_fr = min(frames_planos)
        fibra_min_frame[key] = min_fr
        max_frame_global = max(max_frame_global, max(frames_planos))

    frames_eje_x = []
    fibras_acumuladas_eje_y = []

    for f in range(1, max_frame_global + 1):
        # Contamos cuántas fibras aparecen por primera vez en un frame <= f
        count = sum(1 for fr in fibra_min_frame.values() if fr <= f)
        frames_eje_x.append(f)
        fibras_acumuladas_eje_y.append(count)

    return frames_eje_x, fibras_acumuladas_eje_y


def plot_filtrado_vs_real(datos_filtrado):
    """
    Genera una gráfica con:
      1) Fibras trackeadas acumuladas (JSON filtrado).
      2) Fibras reales por frame (si se encuentra 'fibras_por_frame').
    Anota el valor máximo de cada curva en el gráfico.
    """
    # --- Curva: JSON filtrado (fibras trackeadas acumuladas)
    x_filt, y_filt = compute_fibers_tracked_by_frame(datos_filtrado)

    # --- Curva: "fibras_por_frame" real (por frame)
    x_real, y_real = [], []
    if "fibras_por_frame" in datos_filtrado:
        lista_real = datos_filtrado["fibras_por_frame"]
        x_real = list(range(1, len(lista_real) + 1))
        y_real = lista_real

    plt.figure(figsize=(8, 5))

    # 1) Trackeadas (filtrado)
    plt.plot(x_filt, y_filt, color='red', label='Trackeadas (filtrado)')
    if y_filt:
        max_val_filt = max(y_filt)
        idx_filt = y_filt.index(max_val_filt)
        x_pos_filt = x_filt[idx_filt]
        plt.text(x_pos_filt, max_val_filt,
                 f"max={max_val_filt}",
                 color='red', ha='left', va='bottom', fontsize=9)

    # 2) Fibras reales por frame
    if x_real and y_real:  # Solo si existe la lista
        plt.plot(x_real, y_real, color='green', label='Fibras reales por frame')
        max_val_real = max(y_real)
        idx_real = y_real.index(max_val_real)
        x_pos_real = x_real[idx_real]
        plt.text(x_pos_real, max_val_real,
                 f"max={max_val_real}",
                 color='green', ha='left', va='bottom', fontsize=9)

    # Escala logarítmica en el eje Y (opcional)
    plt.yscale('log')

    plt.title('Comparación de fibras trackeadas (filtrado) y reales')
    plt.xlabel('Frame')
    plt.ylabel('Cantidad de fibras')
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Carga del JSON filtrado
    with open(archivo_fibras_filtrado, "r", encoding="utf-8") as f:
        datos_filt = json.load(f)

    # Generar la gráfica comparativa (con anotación de los máximos)
    plot_filtrado_vs_real(datos_filt)
