import json
import matplotlib.pyplot as plt
import numpy as np

# 1. Cargar el JSON desde la ruta especificada
json_file = "Alpha-Beta-Gamma/fibra_21.json"  # Asegúrate de que la ruta sea la correcta

with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# 2. Extraer la información del JSON
# Se espera que el JSON tenga al menos las siguientes claves:
# "centroide": lista de [x, y] y "frame": lista de [frame_number]
centroides = data["centroide"]
frames = [item[0] for item in data["frame"]]

# Convertir la lista de centroides a un array de NumPy para facilitar los cálculos
centroides = np.array(centroides)  # Forma (N,2)
x = centroides[:, 0]
y = centroides[:, 1]

# 3. Calcular las diferencias (desplazamientos) entre frame y frame
dx = np.diff(x)
dy = np.diff(y)

# 4. Calcular velocidades en x e y
fps = 200.0
dt = 1 / fps  # dt = 0.005 s
vx = dx / dt
vy = dy / dt

# Los frames asociados a las diferencias son desde el segundo valor
frames_v = frames[1:]  

# 5. Función para aplicar una convolución (media móvil) que suavice la señal
def smooth_signal(signal, window_size=5):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(signal, window, mode='same')

# Elegir el tamaño de la ventana (puedes modificarlo según lo necesites)
window_size = 5

vx_smooth = smooth_signal(vx, window_size)
vy_smooth = smooth_signal(vy, window_size)

# 6. Crear los subplots para graficar (en cada gráfico se mostrarán ambas curvas: original y suavizada)
fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Gráfico de velocidad en X
axs[0].plot(frames_v, vx, marker='.', linestyle=':', color='lightblue', label="Original $v_x$")
axs[0].plot(frames_v, vx_smooth, marker='o', linestyle='-', color='blue', label="Suavizada $v_x$")
axs[0].set_ylabel("Velocidad en X (unidades/s)")
axs[0].set_title("Velocidad en X")
axs[0].legend()
axs[0].grid(True)

# Gráfico de velocidad en Y
axs[1].plot(frames_v, vy, marker='.', linestyle=':', color='salmon', label="Original $v_y$")
axs[1].plot(frames_v, vy_smooth, marker='o', linestyle='-', color='red', label="Suavizada $v_y$")
axs[1].set_xlabel("Frame")
axs[1].set_ylabel("Velocidad en Y (unidades/s)")
axs[1].set_title("Velocidad en Y")
axs[1].legend()
axs[1].grid(True)

plt.suptitle("Velocidades (Original vs. Suavizada) a partir de Centroides (fps = 200)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
