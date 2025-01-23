import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# =============================================================================
# 1) LOAD DATA FROM JSON
# =============================================================================
# Specify the JSON file path
json_file = "Graphs/Linear-Velocity/fibers_convolutionated.json"  # Replace with your updated JSON file path

# Load the JSON data
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# =============================================================================
# 2) EXTRACT DATA FOR AVERAGE VELOCITIES
# =============================================================================
# Initialize lists to store data
centroid_x = []
centroid_y = []
vel_x = []
vel_y = []

# Iterate over each fiber in the JSON
for fiber_id, fiber_data in data.items():
    if fiber_id not in ["ruta", "fibras_por_frame"]:  # Skip metadata keys
        centroids = np.array(fiber_data["centroide"])  # Shape (N, 2)
        x = centroids[:, 0]  # X-coordinates
        y = centroids[:, 1]  # Y-coordinates

        # Extract smoothed velocities
        vx_smooth = fiber_data.get("velocidad_x_convolucionada", [])
        vy_smooth = fiber_data.get("velocidad_y_convolucionada", [])

        # Append data for averaging
        centroid_x.extend(x[:len(vx_smooth)])
        centroid_y.extend(y[:len(vy_smooth)])
        vel_x.extend(vx_smooth)
        vel_y.extend(vy_smooth)

# Convert data to NumPy arrays for easier manipulation
centroid_x = np.array(centroid_x)
centroid_y = np.array(centroid_y)
vel_x = np.array(vel_x)
vel_y = np.array(vel_y)

# =============================================================================
# 3) CALCULATE AVERAGE VELOCITIES PER GRID CELL (REDUCED RESOLUTION)
# =============================================================================
# Define the reduced grid size for the heatmap (e.g., 256x256 or 128x128)
grid_size = (128, 128)

# Create 2D histograms for summing velocities and counting fibers
sum_vel_x, x_edges, y_edges = np.histogram2d(
    centroid_x, centroid_y, bins=grid_size, weights=vel_x
)
sum_vel_y, _, _ = np.histogram2d(
    centroid_x, centroid_y, bins=grid_size, weights=vel_y
)
counts, _, _ = np.histogram2d(centroid_x, centroid_y, bins=grid_size)

# Avoid division by zero: calculate average velocities
avg_vel_x = np.divide(sum_vel_x, counts, out=np.zeros_like(sum_vel_x), where=counts > 0)
avg_vel_y = np.divide(sum_vel_y, counts, out=np.zeros_like(sum_vel_y), where=counts > 0)

# Mask cells with no data (counts == 0)
avg_vel_x = np.ma.masked_where(counts == 0, avg_vel_x)
avg_vel_y = np.ma.masked_where(counts == 0, avg_vel_y)

# =============================================================================
# 4) PLOT AVERAGE VELOCITY HEATMAPS
# =============================================================================
# Define the colormap normalization with `TwoSlopeNorm` to center 0 as white
norm_x = TwoSlopeNorm(vmin=np.min(avg_vel_x), vcenter=0, vmax=np.max(avg_vel_x))
norm_y = TwoSlopeNorm(vmin=np.min(avg_vel_y), vcenter=0, vmax=np.max(avg_vel_y))

# Plot average velocity in X
plt.figure(figsize=(10, 8))
plt.imshow(
    avg_vel_x.T,
    origin="upper",  # Ensures (0, 0) is at the top-left
    extent=[0, 1024, 0, 1024],  # Keep the extent for 1024x1024 images
    aspect="auto",
    cmap="coolwarm",
    norm=norm_x
)
plt.colorbar(label="Average Velocity in X (units/s)")
plt.title("Average Velocity in X (Reduced Grid Resolution)")
plt.xlabel("Centroid X Position")
plt.ylabel("Centroid Y Position")
plt.grid(False)
plt.tight_layout()
plt.savefig("Graphs/Linear-Velocity/average_velocity_x.png")
plt.show()

# Plot average velocity in Y
plt.figure(figsize=(10, 8))
plt.imshow(
    avg_vel_y.T,
    origin="upper",  # Ensures (0, 0) is at the top-left
    extent=[0, 1024, 0, 1024],  # Keep the extent for 1024x1024 images
    aspect="auto",
    cmap="coolwarm",
    norm=norm_y
)
plt.colorbar(label="Average Velocity in Y (units/s)")
plt.title("Average Velocity in Y (Reduced Grid Resolution)")
plt.xlabel("Centroid X Position")
plt.ylabel("Centroid Y Position")
plt.grid(False)
plt.tight_layout()
plt.savefig("Graphs/Linear-Velocity/average_velocity_y.png")
plt.show()
