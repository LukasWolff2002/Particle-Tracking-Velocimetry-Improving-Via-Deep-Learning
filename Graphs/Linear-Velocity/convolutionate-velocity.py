import json
import numpy as np

# =============================================================================
# 1) LOAD DATA FROM JSON
# =============================================================================
# Specify the JSON file path
concentracion_fibras = "50"
json_file = f"Particle-Tracking-Velocimetry\\YOLO\\fibras_{concentracion_fibras}_filtrado.json"  # Replace with your actual JSON file path

# Load the JSON data
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# =============================================================================
# 2) DEFINE HELPER FUNCTION: SMOOTH SIGNAL
# =============================================================================
def smooth_signal(signal, window_size=5):
    """
    Smooth a signal using a moving average filter.
    Args:
        signal (np.ndarray): Input signal to be smoothed.
        window_size (int): Size of the moving average window.
    Returns:
        np.ndarray: Smoothed signal.
    """
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(signal, window, mode="same")

# =============================================================================
# 3) PROCESS EACH FIBER
# =============================================================================
# Define the window size for convolution
window_size = 5

# Iterate over each fiber in the JSON
for fiber_id, fiber_data in data.items():
    if fiber_id not in ["ruta", "fibras_por_frame"]:  # Skip metadata keys
        # Extract centroids and calculate velocities
        centroids = np.array(fiber_data["centroide"])  # Shape (N, 2)
        x = centroids[:, 0]  # X-coordinates
        y = centroids[:, 1]  # Y-coordinates

        # Calculate differences (displacements) between consecutive frames
        dx = np.diff(x)
        dy = np.diff(y)

        # Estimate velocities
        fps = 200.0  # Frames per second
        dt = 1 / fps  # Time interval between frames
        vx = dx / dt
        vy = dy / dt

        # Smooth velocities using the moving average filter
        vx_smooth = smooth_signal(vx, window_size)
        vy_smooth = smooth_signal(vy, window_size)

        # Add smoothed velocities back to the fiber data
        # Note: We pad the smoothed arrays to match the number of frames
        fiber_data["velocidad_x_convolucionada"] = vx_smooth.tolist()
        fiber_data["velocidad_y_convolucionada"] = vy_smooth.tolist()

# =============================================================================
# 4) SAVE THE UPDATED JSON
# =============================================================================
# Specify the output JSON file path
output_file = f"Graphs/Linear-Velocity/fibers_{concentracion_fibras}_convolutionated.json"  # Replace with your desired output file name

# Save the updated data to a new JSON file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

print(f"Updated JSON saved to: {output_file}")
