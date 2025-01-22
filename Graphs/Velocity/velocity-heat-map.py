import json
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 1) LOAD DATA FROM JSON
# =============================================================================
# Specify the JSON file path
json_file = "Graphs/Velocity/fibers_convolutionated.json"  # Replace with your updated JSON file path

# Load the JSON data
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# =============================================================================
# 2) EXTRACT DATA FOR HEATMAPS
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

        # Append data for plotting
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
# 3) CREATE HEATMAPS
# =============================================================================
# Define the grid size for the heatmap
grid_size = 50

# Create a 2D histogram for velocity in X
heatmap_x, x_edges, y_edges = np.histogram2d(
    centroid_x, centroid_y, bins=grid_size, weights=vel_x, density=False
)
counts_x, _, _ = np.histogram2d(centroid_x, centroid_y, bins=grid_size)
heatmap_x /= np.maximum(counts_x, 1)  # Avoid division by zero

# Create a 2D histogram for velocity in Y
heatmap_y, _, _ = np.histogram2d(
    centroid_x, centroid_y, bins=grid_size, weights=vel_y, density=False
)
counts_y, _, _ = np.histogram2d(centroid_x, centroid_y, bins=grid_size)
heatmap_y /= np.maximum(counts_y, 1)  # Avoid division by zero

# =============================================================================
# 4) PLOT HEATMAPS
# =============================================================================
# Plot velocity in X
plt.figure(figsize=(10, 8))
plt.imshow(
    heatmap_x.T,
    origin="lower",
    extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    aspect="auto",
    cmap="coolwarm"
)
plt.colorbar(label="Velocity in X (units/s)")
plt.title("Heatmap of Velocity in X")
plt.xlabel("Centroid X Position")
plt.ylabel("Centroid Y Position")
plt.grid(False)
plt.tight_layout()
plt.savefig("heatmap_velocity_x.png")
plt.show()

# Plot velocity in Y
plt.figure(figsize=(10, 8))
plt.imshow(
    heatmap_y.T,
    origin="lower",
    extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
    aspect="auto",
    cmap="coolwarm"
)
plt.colorbar(label="Velocity in Y (units/s)")
plt.title("Heatmap of Velocity in Y")
plt.xlabel("Centroid X Position")
plt.ylabel("Centroid Y Position")
plt.grid(False)
plt.tight_layout()
plt.savefig("heatmap_velocity_y.png")
plt.show()
