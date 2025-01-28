import json
import os
import cv2
import random

# Cargar el diccionario desde el archivo JSON
concentracion_fibras = ["25", "50", "100", "200", "400", "800"]

def graficar (fibras):
    with open(f"C:\\Users\\MBX\\Desktop\\Investigacion\\Particle-Tracking-Velocimetry-Improving-Via-Deep-Learning\\Particle-Tracking-Velocimetry\\Hough-Transform\\fibras_{fibras}_filtrado.json", "r") as file:
        data = json.load(file)

    # Obtener la ruta base de las imágenes (donde se encuentran las imágenes bmp)
    ruta_procesadas = data["ruta"]  # Por ejemplo "runs/segment/predict8"
    ruta_graficos = ruta_procesadas.replace("segment", "graficos")

    # Crear la carpeta para las imágenes con las trayectorias si no existe
    os.makedirs(ruta_graficos, exist_ok=True)

    # Obtener las claves que representan fibras (evitar "ruta")
    fibra_keys = [k for k in data.keys() if k.isdigit()]

    # Determinar el número máximo de fotogramas según el campo "frame"
    max_frame_number = 0
    for fibra_id in fibra_keys:
        frames_list = data[fibra_id]["frame"]
        # Cada elemento de frames_list es, por ejemplo, [1], [2], etc.
        for f in frames_list:
            frame_num = f[0]
            if frame_num > max_frame_number:
                max_frame_number = frame_num

    # Listar las imágenes BMP en la ruta_procesadas
    im_files = [f for f in os.listdir(ruta_procesadas) if f.lower().endswith('.bmp')]
    im_files.sort()

    # Ajustar max_frame_number si hay menos imágenes que frames
    if len(im_files) < max_frame_number:
        print("Advertencia: Hay menos imágenes .bmp que frames en el JSON. Ajustando max_frames al número de imágenes disponibles.")
        max_frame_number = len(im_files)

    # Asignar un color distinto a cada fibra
    colors = {}
    for fibra_id in fibra_keys:
        # Usar el id de la fibra como semilla para generar siempre el mismo color
        random.seed(int(fibra_id))
        b = random.randint(0, 255)
        g = random.randint(0, 255)
        r = random.randint(0, 255)
        colors[fibra_id] = (b, g, r)  # BGR para OpenCV

    # Por cada fotograma, cargar la imagen, dibujar las trayectorias acumuladas y guardarla
    # Ahora frame_idx comienza en 1 porque el "frame" en el JSON comienza en 1, no en 0.
    for frame_idx in range(1, max_frame_number+1):
        img_path = os.path.join(ruta_procesadas, im_files[frame_idx-1])
        img = cv2.imread(img_path)
        if img is None:
            print(f"No se pudo cargar {img_path}. Omitiendo este fotograma.")
            continue

        # Dibujar la trayectoria de cada fibra hasta este frame
        for fibra_id in fibra_keys:
            fibra_data = data[fibra_id]
            centroids = fibra_data["centroide"]
            frames = fibra_data["frame"]  # Lista de listas con el número de frame correspondiente a cada detección
            fiber_color = colors[fibra_id]

            # Filtrar los puntos cuya frame sea <= frame_idx
            coords = []
            for i, f in enumerate(frames):
                if f[0] <= frame_idx:
                    # Agregar el centroid correspondiente si existe
                    c = centroids[i]
                    if len(c) == 2:  # Asegurarnos de que es un punto válido
                        coords.append((c[0], c[1]))

            # Dibujar la trayectoria
            if len(coords) > 1:
                # Dibujar líneas entre puntos consecutivos
                for i in range(len(coords)-1):
                    p1 = (int(coords[i][0]), int(coords[i][1]))
                    p2 = (int(coords[i+1][0]), int(coords[i+1][1]))
                    cv2.line(img, p1, p2, fiber_color, 2)
                # Marcar el último punto con un círculo rojo
                last_p = (int(coords[-1][0]), int(coords[-1][1]))
                cv2.circle(img, last_p, 4, (0, 0, 255), -1)
            elif len(coords) == 1:
                # Solo un punto, dibujar un círculo rojo
                p = (int(coords[0][0]), int(coords[0][1]))
                cv2.circle(img, p, 4, (0, 0, 255), -1)

        # Guardar la imagen modificada
        frame_name = f"frame_{frame_idx:03d}.bmp"
        output_path = os.path.join(ruta_graficos, frame_name)
        cv2.imwrite(output_path, img)

for fibras in concentracion_fibras:
    graficar(fibras)
    
print(f"Imágenes con trayectorias guardadas")