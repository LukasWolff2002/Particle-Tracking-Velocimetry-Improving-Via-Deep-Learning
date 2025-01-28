import cv2
import numpy as np

def detect_lines_and_properties(
    path_imagen,
    roi_points=None,
    canny_threshold1=50,
    canny_threshold2=150,
    hough_threshold=20,
    min_line_length=50,
    max_line_gap=5
):
    """
    Detecta líneas en la imagen dada (usando Canny + HoughLinesP) y retorna:
        - centroids: Lista de centroides [(cx, cy), ...].
        - angles: Lista de ángulos (en grados) de cada línea.
        - lengths: Lista de longitudes de cada línea.
        - scores: Lista de "confianzas" (None, pues HoughLinesP no la provee).
        - boxes: Lista de "cajas" [x1, y1, x2, y2] para cada línea detectada.
    """

    # 1) Cargar la imagen
    imagen = cv2.imread(path_imagen)
    if imagen is None:
        print(f"No se pudo cargar la imagen: {path_imagen}")
        return None, None, None, None, None, None  # agregamos la imagen original (None)

    # 2) Convertir a escala de grises
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # 3) Definir la región de interés (ROI) si se especificó
    if roi_points is not None:
        mask = np.zeros_like(gris)
        cv2.fillPoly(mask, [roi_points], 255)
        gris_roi = cv2.bitwise_and(gris, mask)
    else:
        gris_roi = gris

    # 4) Detectar bordes con Canny
    bordes = cv2.Canny(gris_roi, canny_threshold1, canny_threshold2, apertureSize=3)

    # 5) Detectar líneas con HoughLinesP
    lineas = cv2.HoughLinesP(
        bordes,
        1,
        np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )

    if lineas is None or len(lineas) == 0:
        print("No se detectaron líneas.")
        # Retornamos la imagen cargada para dibujar la ROI, aunque no haya líneas
        return [], [], [], [], [], imagen

    # 6) Preparar listas de salida
    centroids = []
    angles = []
    lengths = []
    scores = []  # HoughLinesP no provee confianza
    boxes = []

    # 7) Calcular propiedades de cada línea
    for linea in lineas:
        x1, y1, x2, y2 = linea[0]

        # Centroide
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # Longitud
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)

        # Ángulo ([-180, 180] en atan2; ajusta según convención)
        angle = np.degrees(np.arctan2(dy, dx))

        centroids.append((cx, cy))
        angles.append(angle)
        lengths.append(length)
        scores.append(None)  # no hay score en HoughLinesP
        boxes.append([x1, y1, x2, y2])

    return centroids, angles, lengths, scores, boxes, imagen


def draw_detections(imagen, roi_points, boxes):
    """
    Dibuja las líneas detectadas (boxes) sobre 'imagen' en color rojo,
    y el contorno de la ROI (si se especifica) en verde.
    Retorna la imagen con los dibujos.
    """

    # Si roi_points no es None, dibujamos la ROI en verde
    if roi_points is not None:
        cv2.polylines(imagen, [roi_points], isClosed=True, color=(0, 255, 0), thickness=2)

    # Dibujamos cada línea en rojo
    for (x1, y1, x2, y2) in boxes:
        cv2.line(imagen, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return imagen


if __name__ == "__main__":
    # Definimos la ROI como polígono, si se desea
    pts = np.array([
        [30, 0],
        [640, 0],
        [640, 840],
        [1024, 840],
        [1024, 980],
        [20, 970]
    ], dtype=np.int32)

    # Ruta de la imagen
    path_imagen = 'Particle-Tracking-Velocimetry/Dataset/25 Fibras/Cam 1/Basler_acA1440-220uc__40343408__20250123_154326693_0063.bmp'

    # Detectamos las líneas y propiedades
    centroids, angles, lengths, scores, boxes, imagen_original = detect_lines_and_properties(
        path_imagen,
        roi_points=pts,
        canny_threshold1=100,
        canny_threshold2=250,
        hough_threshold=40,
        min_line_length=30,
        max_line_gap=5
    )

    # Si la imagen no se pudo cargar, detenemos
    if imagen_original is None:
        exit()

    # (Opcional) Imprimir algunas de las listas
    print("Centroids:", centroids)
    print("Angles:", angles)
    print("Lengths:", lengths)
    # ...

    # Dibujamos las detecciones sobre la imagen original
    # (Si no hubo líneas, boxes estará vacío y no pintará nada)
    imagen_con_lineas = draw_detections(imagen_original, pts, boxes)

    # Mostramos la imagen final en una ventana
    cv2.imshow("Detecciones", imagen_con_lineas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
