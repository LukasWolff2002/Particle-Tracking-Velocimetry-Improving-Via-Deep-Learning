import json
import matplotlib.pyplot as plt
import os

# Lista de concentraciones a comparar
concentraciones = ["25", "50", "100", "200", "400", "800"]

# Directorio base donde estén los JSON filtrados
base_dir = "Particle-Tracking-Velocimetry/YOLO"

def compute_fibers_tracked_by_frame(datos):
    """
    Retorna (frames_eje_x, fibras_acumuladas_eje_y), donde:
      - frames_eje_x = [1, 2, ..., max_frame_en_datos]
      - fibras_acumuladas_eje_y[i] = cuántas fibras han aparecido
        en frame <= frames_eje_x[i].
    """
    fibra_min_frame = {}
    max_frame_global = 0

    for key, value in datos.items():
        if key in ("ruta", "fibras_por_frame"):
            continue

        lista_frames = value.get("frame", [])
        if not lista_frames:
            continue

        frames_planos = [f[0] for f in lista_frames]
        min_fr = min(frames_planos)
        fibra_min_frame[key] = min_fr
        max_frame_global = max(max_frame_global, max(frames_planos))

    frames_eje_x = []
    fibras_acumuladas_eje_y = []

    for f in range(1, max_frame_global + 1):
        count = sum(1 for fr in fibra_min_frame.values() if fr <= f)
        frames_eje_x.append(f)
        fibras_acumuladas_eje_y.append(count)

    return frames_eje_x, fibras_acumuladas_eje_y

def get_normalized_track_curve(concentracion):
    """
    Lee el archivo filtrado de la concentración dada.
    Retorna la curva de fibras trackeadas acumuladas y normalizadas
    (x_filt, y_filt_norm), donde se divide cada valor por el número total de fibras.
    """
    archivo_fibras_filtrado = os.path.join(base_dir, f"fibras_{concentracion}_filtrado.json")
    total_fibras = int(concentracion)  # Asumimos que la concentración es el número real de fibras

    with open(archivo_fibras_filtrado, "r", encoding="utf-8") as f:
        datos_filtrado = json.load(f)

    x_filt, y_filt = compute_fibers_tracked_by_frame(datos_filtrado)
    # Normalizar
    y_filt_norm = [val / total_fibras for val in y_filt]
    return x_filt, y_filt_norm

# =============================================================================
# GRAFICAR LAS CURVAS DE TRAKEO (NORMALIZADAS) EN UN SOLO GRÁFICO
# =============================================================================

plt.figure(figsize=(10, 6))

# Paleta de colores "Dark2" en formato HEX para 6 concentraciones
dark2_colors = [
    "#1B9E77",  # verde azulado
    "#D95F02",  # naranja
    "#7570B3",  # morado
    "#E7298A",  # rosa fuerte
    "#66A61E",  # verde
    "#E6AB02",  # ocre
]

for i, conc in enumerate(concentraciones):
    x_filt, y_filt_norm = get_normalized_track_curve(conc)

    # Asignamos un color distinto de la lista dark2_colors
    color = dark2_colors[i % len(dark2_colors)]
    
    # Graficamos la curva (mismo estilo de la versión anterior)
    plt.plot(x_filt, y_filt_norm, color=color, label=conc)

plt.yscale("log")  # Se mantiene la escala logarítmica en el eje Y

plt.title("Curvas Normalizadas de Fibras Trackeadas (Filtrado)")
plt.xlabel("Frame")
plt.ylabel("Fracción de fibras (trackeadas / total_fibras)")
plt.grid(True)
plt.legend(title="Concentración", loc="best")
plt.tight_layout()
plt.savefig("Graphs/YOLO/Eficiencia/Graphs/tracked_fibers_normalized.png")
