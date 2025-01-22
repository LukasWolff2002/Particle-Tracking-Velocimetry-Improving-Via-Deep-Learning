import cv2
import numpy as np

# Cargar la imagen
imagen = cv2.imread('Dataset\Toma2\Basler_acA1440-220uc__40343408__20250108_121135810_0000.bmp')
if imagen is None:
    print("No se pudo cargar la imagen. Revisa la ruta.")
    exit()

# Convertir a escala de grises
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

# Definir los 6 puntos de la región de interés (ROI)
pts = np.array([
    [0, 0],
    [380, 0],
    [385, 800],
    [1024, 810],
    [1024, 980],
    [0, 980]
], dtype=np.int32)

# Crear una máscara (mismo tamaño que la imagen en gris)
mask = np.zeros_like(gris)
# Rellenar el polígono con color blanco (255)
cv2.fillPoly(mask, [pts], 255)

# Aplicar la máscara a la imagen gris
roi = cv2.bitwise_and(gris, mask)

# Detectar bordes con Canny dentro de la ROI
bordes = cv2.Canny(roi, 50, 150, apertureSize=3)

# Detectar líneas con HoughLinesP
lineas = cv2.HoughLinesP(
    bordes, 
    1, 
    np.pi / 180, 
    threshold=20, 
    minLineLength=50, 
    maxLineGap=5
)

# Dibujar las líneas detectadas en la imagen original (en rojo)
if lineas is not None:
    for linea in lineas:
        x1, y1, x2, y2 = linea[0]
        cv2.line(imagen, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Dibujar el contorno del polígono de la ROI (en verde)
cv2.polylines(imagen, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

# Mostrar únicamente la imagen final con el contorno de la ROI y las líneas detectadas
cv2.imshow('Imagen Final', imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()
