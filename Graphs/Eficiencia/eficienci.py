import json
import matplotlib.pyplot as plt
import os

# Lista de concentraciones a comparar
concentraciones = ["25", "50", "100", "200", "400", "800"]

# Directorio base donde estén los JSON filtrados
# Ajusta esta variable a tu estructura de carpetas si es necesario.
base_dir = "Particle-Tracking-Velocimetry\\YOLO"

def compute_fibers_tracked_by_frame(datos):
    """
    Dado un diccionario (contenido de un JSON filtrado),
    retorna (frames_eje_x, fibras_acumuladas_eje_y).
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

def get_normalized_curves(concentracion, total_fibras):
    """
    Carga el archivo filtrado para 'concentracion'.
    Retorna dos curvas (x_filt, y_filt_norm) y (x_real, y_real_norm),
    donde ambas están normalizadas dividiendo por 'total_fibras'.
    Si 'fibras_por_frame' no existe en el JSON, la curva real será vacía.
    """
    archivo_fibras_filtrado = os.path.join(base_dir, f"fibras_{concentracion}_filtrado.json")

    with open(archivo_fibras_filtrado, "r", encoding="utf-8") as f:
        datos_filtrado = json.load(f)

    # Curva filtrada (trackeadas acumuladas, normalizadas)
    x_filt, y_filt = compute_fibers_tracked_by_frame(datos_filtrado)
    y_filt_norm = [val / total_fibras for val in y_filt]

    # Curva real (por frame, normalizada)
    x_real, y_real_norm = [], []
    if "fibras_por_frame" in datos_filtrado:
        lista_real = datos_filtrado["fibras_por_frame"]
        x_real = list(range(1, len(lista_real) + 1))
        y_real_norm = [val / total_fibras for val in lista_real]

    return x_filt, y_filt_norm, x_real, y_real_norm

# =============================================================================
# GRAFICAR TODAS LAS CURVAS (TRACK Y REAL) EN UN SOLO GRÁFICO
# =============================================================================

plt.figure(figsize=(10, 6))

# Para asignar un color distinto a cada concentración
color_cycle = plt.cm.get_cmap("tab10", len(concentraciones))
# (La mayoría de colormaps admiten hasta 10 colores distintos, "tab10" es común.
#  Si tienes más de 10 concentraciones, cambiar a un colormap que soporte más colores.)

for i, conc in enumerate(concentraciones):
    # Convertir a entero si el número de fibras en 'conc' representa la cantidad real
    # o ajustarlo si quieres un valor fijo. Supongamos que 'conc' = "800" => 800
    total_fibras = int(conc)

    # Obtenemos las curvas normalizadas
    x_filt, y_filt_norm, x_real, y_real_norm = get_normalized_curves(conc, total_fibras)

    color = color_cycle(i)  # color distinto para cada concentración

    # Curva de trackeadas
    plt.plot(x_filt, y_filt_norm, color=color,
             label=f"Track (filtrado) - {conc}")

    # Curva de reales (si existe)
    if x_real and y_real_norm:
        plt.plot(x_real, y_real_norm, "--", color=color,
                 label=f"Real - {conc}")  # guión discont. p/distinción

# Escala logarítmica en el eje Y (opcional)
plt.yscale('log')

plt.title("Comparación Normalizada de Fibras Trackeadas (filtrado) vs. Reales\n"
          "Todas las concentraciones en un mismo gráfico")
plt.xlabel("Frame")
plt.ylabel("Fracción de fibras (valor / total_fibras)")
plt.grid(True)
plt.legend(loc="best")
plt.tight_layout()

plt.show()
