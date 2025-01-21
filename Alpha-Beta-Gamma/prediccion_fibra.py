import json
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
# Ruta al archivo JSON
ruta_archivo = 'PTV/INFORME_5/CODIGO/fibra_21.json'

# Abrir y leer el archivo JSON
with open(ruta_archivo, 'r', encoding='utf-8') as archivo:
    datos = json.load(archivo)


centroides = datos['centroide']
largos = datos['largo_maximo']
angulos = datos['angulo']
frame = datos['frame']

alpha = 0.2
betha = 0.2
gamma = 0.2


import numpy as np
import matplotlib.pyplot as plt
import os

def graficar(centroide_real, largo_real, angulo_real, frame,
            centroide_prediccion, largo_prediccion, angulo_prediccion,
            centroide_real_2, largo_real_2, angulo_real_2,
            tamaño=(1024, 1024), color_real='green',
            color_prediccion='blue', grosor=3,
            directorio='PTV/INFORME_5/GRAFICOS/PREDICCIONES',
            formato='jpg', calidad_jpeg=90):

    
    ancho, alto = tamaño
    
    # Crear el directorio si no existe
    os.makedirs(directorio, exist_ok=True)
    
    # Crear la figura y el eje
    dpi = 100  # Dots per inch
    fig, ax = plt.subplots(figsize=(ancho / dpi, alto / dpi), dpi=dpi)
    
    # Configurar límites y aspecto
    ax.set_xlim(0, ancho)
    ax.set_ylim(0, alto)
    ax.set_aspect('equal')
    ax.axis('off')  # Ocultar ejes
    
    # Invertir el eje Y para alinear con OpenCV
    ax.invert_yaxis()
    
    # Función interna para calcular coordenadas de la línea
    def calcular_linea(centroide, largo, angulo):
        # Convertir el ángulo a radianes
        angulo_rad = np.deg2rad(angulo)

        # Calcular las diferencias en X e Y basadas en el ángulo
        delta_x = (largo / 2) * np.cos(angulo_rad)
        delta_y = (largo / 2) * np.sin(angulo_rad)

        # Calcular los puntos inicial y final de la línea
        x_inicio = centroide[0] - delta_x
        y_inicio = centroide[1] - delta_y
        x_fin = centroide[0] + delta_x
        y_fin = centroide[1] + delta_y

        return (x_inicio, y_inicio), (x_fin, y_fin)

    
    # Calcular coordenadas de las líneas
    inicio_real, fin_real = calcular_linea(centroide_real, largo_real, angulo_real)
    inicio_prediccion, fin_prediccion = calcular_linea(centroide_prediccion, largo_prediccion, angulo_prediccion)
    inicio_real_2, fin_real_2 = calcular_linea(centroide_real_2, largo_real_2, angulo_real_2)
    
    # Dibujar las líneas
    ax.plot([inicio_real[0], fin_real[0]], [inicio_real[1], fin_real[1]], color=color_real, linewidth=grosor, label='Real')
    ax.plot([inicio_prediccion[0], fin_prediccion[0]], [inicio_prediccion[1], fin_prediccion[1]], color=color_prediccion, linewidth=grosor, label='Predicción')
    ax.plot([inicio_real_2[0], fin_real_2[0]], [inicio_real_2[1], fin_real_2[1]], color='red', linewidth=grosor, label='Real frame prediccion')
    
    # Añadir la leyenda
    ax.legend(loc='upper right', fontsize=10, frameon=True)

    fig.suptitle(f'Parametros {alpha=},{betha=},{gamma=}', fontsize=16, y=0.95)
    
    # Ajustar la figura para eliminar espacios
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    subdirectorio = f'alpha={alpha}-betha={betha}-gamma={gamma}'
    directorio_completo = os.path.join(directorio, subdirectorio)

    # Crear el subdirectorio si no existe
    os.makedirs(directorio_completo, exist_ok=True)

    
    # Definir la ruta de guardado con el formato seleccionado
    ruta_guardado = os.path.join(directorio_completo, f'imagen_fibra_{frame}.{formato}')
    
    # Configurar parámetros de compresión según el formato
    if formato.lower() in ['jpg', 'jpeg']:
        params = {'quality': calidad_jpeg, 'optimize': True, 'progressive': True}
    elif formato.lower() == 'webp':
        params = {'quality': calidad_jpeg}
    else:
        params = {}
    
    # Guardar la figura utilizando Matplotlib
    try:
        plt.savefig(ruta_guardado, format=formato, bbox_inches='tight', pad_inches=0, **params)
    except TypeError as e:
        print(f"Advertencia: {e}. Guardando sin parámetros adicionales.")
        plt.savefig(ruta_guardado, format=formato, bbox_inches='tight', pad_inches=0)
    
    # Cerrar la figura para liberar memoria
    plt.close(fig)
    
    print(f"Imagen guardada en {ruta_guardado}")









centroides_prediccion = []
velocidades_prediccion = []
aceleraciones_prediccion = []

angulo_xy_prediccion = []
velocidad_angular_xy_prediccion = []
aceleracion_angular_xy_prediccion = []

fps = 200
delta_t = 1/fps


for i in range(len(centroides)):
    
    if i == 0:
       
        #Estoy en la primera deteccion
        #Por lo tanto no tengo una prediccion, solo tomo los valores reales

        
        x_0 = centroides[i][0]
        y_0 = centroides[i][1]
        v_x_0 = 0
        v_y_0 = 0
        a_x_0 = 0
        a_y_0 = 0
        angulo_0 = angulos[i][0]
        v_angular_xy_0 = 0
        a_angular_xy_0 = 0
        largo_0 = largos[i][0]
    

        #Ahora si debo generar una prediccion
        x_1 = (x_0) + (v_x_0*delta_t) + (0.5 * a_x_0 * delta_t**2)
        y_1 = (y_0) + (v_y_0*delta_t) + (0.5 * a_y_0 * delta_t**2)
        centroides_prediccion.append([x_1, y_1])

        v_x_1 = v_x_0 + (a_x_0 * delta_t)
        v_y_1 = v_y_0 + (a_y_0 * delta_t)
        velocidades_prediccion.append([v_x_1, v_y_1])

        a_x_1 = a_x_0
        a_y_1 = a_y_0
        aceleraciones_prediccion.append([a_x_1, a_y_1])

        angulo_1 = angulo_0 + v_angular_xy_0 * delta_t + 0.5 * a_angular_xy_0 * delta_t**2
        angulo_xy_prediccion.append(angulo_1)

        velocidad_angular_xy_1 = v_angular_xy_0 + a_angular_xy_0 * delta_t
        velocidad_angular_xy_prediccion.append(velocidad_angular_xy_1)

        aceleracion_angular_xy_1 = a_angular_xy_0
        aceleracion_angular_xy_prediccion.append(aceleracion_angular_xy_1)
        

    else:

        #Ahora detecto una fibra segun la prediccion anterior
        z_x = centroides[i][0]
        z_y = centroides[i][1]
        z_angulo = angulos[i][0]
        z_largo = largos[i][0]

        #Genero mis ecuaciones de estado

        x_1 = x_1 + alpha * (z_x - x_1)
        y_1 = y_1 + alpha * (z_y - y_1)

        v_x_1 = v_x_1 + betha * ((z_x - x_1)/delta_t)
        v_y_1 = v_y_1 + betha * ((z_y - y_1)/delta_t)

        a_x_1 = a_x_1 + gamma * ((z_x - x_1)/2*delta_t**2)
        a_y_1 = a_y_1 + gamma * ((z_y - y_1)/2*delta_t**2)

        angulo_1 = angulo_1 + alpha * (z_angulo - angulo_1)

        velocidad_angular_xy_1 = velocidad_angular_xy_1 + betha * ((z_angulo - angulo_1)/delta_t)

        aceleracion_angular_xy_1 = aceleracion_angular_xy_1 + gamma * ((z_angulo - angulo_1)/2*delta_t**2)

        #Ahora genero la prediccion

        x_2 = x_1 + v_x_1 * delta_t + 0.5 * a_x_1 * delta_t**2
        y_2 = y_1 + v_y_1 * delta_t + 0.5 * a_y_1 * delta_t**2

        v_x_2 = v_x_1 + a_x_1 * delta_t
        v_y_2 = v_y_1 + a_y_1 * delta_t

        a_x_2 = a_x_1
        a_y_2 = a_y_1

        angulo_2 = angulo_1 + velocidad_angular_xy_1 * delta_t + 0.5 * aceleracion_angular_xy_1 * delta_t**2
        
        velocidad_angular_xy_2 = velocidad_angular_xy_1 + aceleracion_angular_xy_1 * delta_t

        aceleracion_angular_xy_2 = aceleracion_angular_xy_1

        #Guardo las predicciones

        centroides_prediccion.append([x_2, y_2])
        velocidades_prediccion.append([v_x_2, v_y_2])
        aceleraciones_prediccion.append([a_x_2, a_y_2])
        angulo_xy_prediccion.append(angulo_2)
        velocidad_angular_xy_prediccion.append(velocidad_angular_xy_2)
        aceleracion_angular_xy_prediccion.append(aceleracion_angular_xy_2)

        #Actualizo los valores para la siguiente iteracion

        x_1 = x_2
        y_1 = y_2

        v_x_1 = v_x_2
        v_y_1 = v_y_2

        a_x_1 = a_x_2
        a_y_1 = a_y_2

        angulo_1 = angulo_2
        
        velocidad_angular_xy_1 = velocidad_angular_xy_2

        aceleracion_angular_xy_1 = aceleracion_angular_xy_2


#Ahora puedo graficar las predicciones

#for i in range(len(centroides)-1):
    #graficar(centroides[i], largos[i][0], angulos[i][0], frame[i], centroides_prediccion[i+1], largos[i+1][0], angulo_xy_prediccion[i+1], centroides[i+1], largos[i+1][0], angulos[i+1][0])


velocidades_ptv = []
velocidades_prediccion = velocidades_prediccion[1:]
frame = frame[1:]

for i in range(len(centroides)-1):
    velocidad_x = ((centroides[i+1][0] - centroides[i][0])/delta_t)
    velocidad_y = (centroides[i+1][1] - centroides[i][1])/delta_t

    velocidades_ptv.append([velocidad_x, velocidad_y])
  

# Ahora puedo graficar las diferencias
# Separar las velocidades en ejes X e Y para real y predicción
velocidades_ptv = np.array(velocidades_ptv)

velocidad_ptv_x = velocidades_ptv[:, 0]
velocidad_ptv_y = velocidades_ptv[:, 1]

velocidades_prediccion = np.array(velocidades_prediccion)

velocidad_prediccion_x = velocidades_prediccion[:, 0]
velocidad_prediccion_y = velocidades_prediccion[:, 1]

# Calcular las diferencias entre real y predicción para ambos ejes
diferencia_velocidad_x = velocidad_ptv_x - velocidad_prediccion_x
diferencia_velocidad_y = velocidad_ptv_y - velocidad_prediccion_y

# Crear una figura con tres subplots (uno para X, otro para Y, y otro para las diferencias)
fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

plt.suptitle(f"Velocidades: Real vs Predicción (α={alpha}, β={betha}, γ={gamma})", fontsize=16)

# **Primer Subplot: Velocidad X Real vs Predicción**
axs[0].plot(frame, velocidad_ptv_x, label="Velocidad Real X", marker='o', linestyle='-', color='blue')
axs[0].plot(frame, velocidad_prediccion_x, label="Velocidad Predicción X", marker='x', linestyle='--', color='red')
axs[0].set_title("Velocidades en el eje X")
axs[0].set_ylabel("Velocidad X (px/s)")
axs[0].legend()
axs[0].grid(True)

# **Segundo Subplot: Velocidad Y Real vs Predicción**
axs[1].plot(frame, velocidad_ptv_y, label="Velocidad Real Y", marker='o', linestyle='-', color='blue')
axs[1].plot(frame, velocidad_prediccion_y, label="Velocidad Predicción Y", marker='x', linestyle='--', color='red')
axs[1].set_title("Velocidades en el eje Y")
axs[1].set_ylabel("Velocidad Y (px/s)")
axs[1].legend()
axs[1].grid(True)



# Ajustar el espacio entre subplots para evitar solapamientos
plt.tight_layout()



# Definir la ruta de guardado
#ruta_guardado = os.path.join(directorio_guardado, 'diferencias_velocidad.png')

# Guardar la figura sin bordes ni márgenes adicionales
plt.savefig(f'PTV/INFORME_5/INFORME/GRAFICOS/VELOCIDADES/alpha_{alpha}_betha_{betha}_gamma_{gamma}.png', bbox_inches='tight', pad_inches=0.1)

# Mostrar la figura (opcional)
#plt.show()

# Cerrar la figura para liberar memoria
#plt.close(fig)

#print(f"Gráfica guardada en {ruta_guardado}")