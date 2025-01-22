import re
import csv

# Regex para extraer la información de cada línea de entrenamiento donde aparece:
# "Train Epoch: [EPOCH][  IT/IT_TOTAL] | ... | Losses/train_all_loss: VALOR (PROMEDIO)"
# 
# Ejemplo de línea:
# INFO 2025-01-21 13:26:19,065 train_utils.py: 271: Train Epoch: [0][  0/609] | ... | Losses/train_all_loss: 3.73e+00 (3.73e+00)

regex_line = re.compile(
    r"Train Epoch: \[(\d+)\]\[\s*(\d+)/(\d+)\].*Losses/train_all_loss:\s*([0-9\.e\+\-]+)\s*\(([0-9\.e\+\-]+)\)"
)

# Cambia esta ruta al archivo .txt que contiene tus logs:
logfile = "Segmentation-Models\Training-Results\sam2.txt"

# Lista donde almacenaremos los datos extraídos
data = []

with open(logfile, "r", encoding="utf-8") as f:
    for line in f:
        match = regex_line.search(line)
        if match:
            epoch = int(match.group(1))
            iteration = int(match.group(2))
            total_iter = int(match.group(3))
            current_loss = float(match.group(4))   # Valor actual de la pérdida
            avg_loss = float(match.group(5))       # Valor promedio de la pérdida

            data.append((epoch, iteration, total_iter, current_loss, avg_loss))

# Si quieres guardar en CSV
csv_file = "Segmentation-Models\Training-Results\sam2_losses.csv"
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # Cabecera
    writer.writerow(["epoch", "iteration", "iteration_total", "current_loss", "avg_loss"])
    # Filas
    writer.writerows(data)

print(f"Se extrajeron {len(data)} filas de datos. Guardado en {csv_file}.")
