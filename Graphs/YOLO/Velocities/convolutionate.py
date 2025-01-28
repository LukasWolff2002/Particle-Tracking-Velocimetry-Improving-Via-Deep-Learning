import json
import numpy as np

# =============================================================================
# 1) PARÁMETROS INICIALES
# =============================================================================

# Frames por segundo (para calcular dt)
fps = 200.0
dt = 1.0 / fps

# Tamaño de la ventana de suavizado por convolución
window_size = 5

# =============================================================================
# 2) FUNCIONES AUXILIARES
# =============================================================================

def smooth_signal(signal, window_size=5):
    """
    Suaviza una señal usando un filtro de media móvil.
    
    Args:
        signal (np.ndarray): Señal de entrada a suavizar.
        window_size (int): Tamaño de la ventana de la media móvil.
        
    Returns:
        np.ndarray: Señal suavizada.
    """
    # Si la señal es muy corta, se retorna tal cual para evitar errores
    if len(signal) < 2:
        return signal

    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(signal, window, mode="same")

def angular_difference(angle1, angle2):
    """
    Calcula la diferencia angular más pequeña entre angle1 y angle2,
    resultando en un valor en el rango [-180, 180].
    
    Args:
        angle1 (float): Primer ángulo en grados.
        angle2 (float): Segundo ángulo en grados.
        
    Returns:
        float: Diferencia angular en grados.
    """
    diff = angle1 - angle2
    while diff > 180:
        diff -= 360
    while diff <= -180:
        diff += 360
    return diff

# =============================================================================
# 3) FUNCIÓN PRINCIPAL: PROCESAMIENTO Y GUARDADO
# =============================================================================

def convolutionated(fibras):
    """
    Lee un JSON de fibras, calcula y suaviza (por convolución) 
    tanto la velocidad lineal como la velocidad angular de cada fibra,
    y finalmente guarda un nuevo JSON con esos valores.
    """

    # ----------------------------------------------------------------------------
    # 3.1) CARGA DE DATOS
    # ----------------------------------------------------------------------------
    json_file = f"Particle-Tracking-Velocimetry\\YOLO\\fibras_{fibras}_filtrado.json"
    output_file = f"Graphs/YOLO/Velocities/fibers_{fibras}_convolutionated.json"
    
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ----------------------------------------------------------------------------
    # 3.2) PROCESAMIENTO DE CADA FIBRA: VELOCIDAD LINEAL Y ANGULAR
    # ----------------------------------------------------------------------------
    for fiber_id, fiber_data in data.items():
        # Omitimos claves que no son fibras
        if fiber_id in ["ruta", "fibras_por_frame"]:
            continue
        
        # -------------------------
        # Velocidad lineal
        # -------------------------
        if "centroide" in fiber_data:
            centroids = np.array(fiber_data["centroide"])  # cada elemento es [x, y]
            if centroids.shape[0] > 1:
                x = centroids[:, 0]
                y = centroids[:, 1]
                
                # Diferencias entre frames consecutivos
                dx = np.diff(x)
                dy = np.diff(y)
                
                # Velocidades
                vx = dx / dt
                vy = dy / dt
                
                # Suavizado
                vx_smooth = smooth_signal(vx, window_size)
                vy_smooth = smooth_signal(vy, window_size)
                
                # Guardamos en el JSON como listas
                fiber_data["velocidad_x_convolucionada"] = vx_smooth.tolist()
                fiber_data["velocidad_y_convolucionada"] = vy_smooth.tolist()
            else:
                # Si no hay suficiente longitud para calcular diffs
                fiber_data["velocidad_x_convolucionada"] = []
                fiber_data["velocidad_y_convolucionada"] = []
        
        # -------------------------
        # Velocidad angular
        # -------------------------
        if "angulo" in fiber_data:
            # Asumiendo que "angulo" es lista de listas: [ [angulo1], [angulo2], ... ]
            angles_raw = fiber_data["angulo"]
            if angles_raw and isinstance(angles_raw[0], list):
                # Extraemos solo el primer valor de cada sublista como ángulo
                angles = np.array([item[0] for item in angles_raw])
            else:
                # O si "angulo" es simplemente una lista de valores
                angles = np.array(angles_raw)
            
            if len(angles) > 1:
                angular_velocities = [
                    angular_difference(angles[i + 1], angles[i]) / dt
                    for i in range(len(angles) - 1)
                ]
                angular_velocities = np.array(angular_velocities)
                
                # Suavizado
                angular_velocities_smooth = smooth_signal(angular_velocities, window_size)
                
                # Guardamos en el JSON
                fiber_data["velocidad_angular_convolucionada"] = angular_velocities_smooth.tolist()
            else:
                fiber_data["velocidad_angular_convolucionada"] = []
    
    # ----------------------------------------------------------------------------
    # 3.3) GUARDADO DE RESULTADOS
    # ----------------------------------------------------------------------------
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"Archivo JSON actualizado y guardado en: {output_file}")

# =============================================================================
# 4) BUCLE DE EJECUCIÓN SOBRE DIFERENTES CONCENTRACIONES
# =============================================================================

concentracion_fibras = ["25", "50", "100", "200", "400", "800"]

for fibras in concentracion_fibras:
    convolutionated(fibras)
