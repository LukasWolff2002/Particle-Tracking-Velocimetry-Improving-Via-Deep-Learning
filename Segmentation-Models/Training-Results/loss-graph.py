import csv
import matplotlib.pyplot as plt
from collections import defaultdict

#### 1) Procesar CSV de SAM ####
# Archivo que contiene las métricas de SAM:
csv_file_sam = "Segmentation-Models/Training-Results/sam2_losses.csv"

# Usamos diccionarios para acumular la suma y el conteo de 'avg_loss' por época.
sam_epoch_sums = defaultdict(float)
sam_epoch_counts = defaultdict(int)

with open(csv_file_sam, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        epoch = int(row["epoch"])
        avg_loss = float(row["avg_loss"])
        sam_epoch_sums[epoch] += avg_loss
        sam_epoch_counts[epoch] += 1

# Ordenamos las épocas
sam_epochs = sorted(sam_epoch_sums.keys())
# Calculamos el valor promedio de loss por época para SAM
sam_avg_losses = [sam_epoch_sums[e] / sam_epoch_counts[e] for e in sam_epochs]

#### 2) Procesar CSV de YOLO ####
# Archivo que contiene las métricas de YOLO:
csv_file_yolo = "Segmentation-Models/Training-Results/yolo.csv"

# Queremos agrupar por época y calcular para cada época la suma de
# train/box_loss + train/seg_loss + train/cls_loss + train/dfl_loss.
# En este ejemplo asumiremos que cada fila es de una época.
# (Si hay más de una fila por época, se puede agrupar de forma similar a SAM).

yolo_epochs = []
yolo_total_losses = []

with open(csv_file_yolo, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            epoch = int(row["epoch"])
            # Convertimos cada loss a float y sumamos
            loss = float(row["train/cls_loss"])
            total_loss = loss 
        except ValueError:
            # En caso de que alguna celda contenga 'inf' u otro valor no convertible,
            # se puede omitir la fila o asignar un valor muy alto.
            continue
        
        yolo_epochs.append(epoch)
        yolo_total_losses.append(total_loss)

# Si hay múltiples registros por época en YOLO y deseas agruparlos,
# puedes usar un enfoque similar al de SAM.
# (Aquí se asume que cada línea corresponde a una época).

#### 3) Graficar ambas curvas (loss) ####

plt.figure(figsize=(10,6))

# Curva SAM: pérdida promedio por época
plt.plot(sam_epochs, sam_avg_losses, marker='o', linestyle='-', label="SAM")

# Curva YOLO: total de loss de entrenamiento por época
plt.plot(yolo_epochs, yolo_total_losses, marker='x', linestyle='-', label="YOLO")

plt.xlabel("Época")
plt.ylabel("Valor de Loss")
plt.title("Comparación de Curvas de Loss: SAM vs YOLO")
plt.legend()
plt.grid(True)
plt.savefig("Segmentation-Models/Training-Results/loss_graph.png")
