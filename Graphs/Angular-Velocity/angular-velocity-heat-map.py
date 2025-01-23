import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# =============================================================================
# 1) LOAD DATA FROM JSON
# =============================================================================
# Specify the JSON file path
json_file = "Graphs/Angular-Velocity/fiber_convolutionated.json"  # Replace with your JSON file path

# Load the JSON data
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# =============================================================================
# 2) EXTRACT DATA FOR AVERAGE ANGULAR VELOCITIES
# =============================================================================
# Initialize lists to store data
centroid_x = []
centroid_y = []
angular_vel = []

# Iterate over each fiber in the JSON
for fiber_id, fiber_data in data.items():
    if fiber_id not in ["ruta", "fibras_por_frame"]:  # Skip metadata keys
        centroids = np.array(fiber_data["centroide"])  # Shape (N, 2)
        x = centroids[:, 0]  # X-coordinates
        y = centroids[:, 1]  # Y-coordinates

        # Extract smoothed angular velocities
        angular_vel_smooth = fiber_data.get("velocidad_angular_convolucionada", [])

        # Append data for averaging
        centroid_x.extend(x[:len(angular_vel_smooth)])
        centroid_y.extend(y[:len(angular_vel_smooth)])
        angular_vel.extend(angular_vel_smooth)

# Convert data to NumPy arrays for easier manipulation
centroid_x = np.array(centroid_x)
centroid_y = np.array(centroid_y)
angular_vel = np.array(angular_vel)

# =============================================================================
# 3) CALCULATE AVERAGE ANGULAR VELOCITIES PER GRID CELL (REDUCED RESOLUTION)
# =============================================================================
# Define the reduced grid size for the heatmap (e.g., 128x128 or 256x256)
grid_size = (128, 128)

# Create 2D histograms for summing angular velocities and counting fibers
sum_angular_vel, x_edges, y_edges = np.histogram2d(
    centroid_x, centroid_y, bins=grid_size, weights=angular_vel
)
counts, _, _ = np.histogram2d(centroid_x, centroid_y, bins=grid_size)

# Avoid division by zero: calculate average angular velocities
avg_angular_vel = np.divide(
    sum_angular_vel, counts, out=np.zeros_like(sum_angular_vel), where=counts > 0
)

# Mask cells with no data (counts == 0)
avg_angular_vel = np.ma.masked_where(counts == 0, avg_angular_vel)

# =============================================================================
# 4) PLOT AVERAGE ANGULAR VELOCITY HEATMAP
# =============================================================================
# Define the colormap normalization with `TwoSlopeNorm` to center 0 as white
norm = TwoSlopeNorm(vmin=np.min(avg_angular_vel), vcenter=0, vmax=np.max(avg_angular_vel))

# Plot average angular velocity
plt.figure(figsize=(10, 8))
plt.imshow(
    avg_angular_vel.T,
    origin="upper",  # Ensures (0, 0) is at the top-left
    extent=[0, 1024, 0, 1024],  # Keep the extent for 1024x1024 images
    aspect="auto",
    cmap="coolwarm",
    norm=norm
)
plt.colorbar(label="Average Angular Velocity (degrees/s)")
plt.title("Average Angular Velocity (Reduced Grid Resolution)")
plt.xlabel("Centroid X Position")
plt.ylabel("Centroid Y Position")
plt.grid(False)
plt.tight_layout()
#plt.savefig("Graphs/Velocity/average_angular_velocity_reduced.png")
plt.show()
