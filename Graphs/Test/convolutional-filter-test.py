import json
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# 1) GLOBAL STYLE SETTINGS
# =============================================================================
# Configure global plot styles for consistency and readability
import matplotlib
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.2,       # Border line width
    "lines.linewidth": 2.0,      # Main line width
})

# =============================================================================
# 2) LOAD DATA FROM JSON
# =============================================================================
# Specify the path to the JSON file
json_file = "Alpha-Beta-Gamma/fibra_21.json"  # Ensure the path is correct

# Load the JSON data
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract data from JSON
# Expected keys: "centroide" (list of [x, y]) and "frame" (list of [frame_number])
centroids = data["centroide"]
frames = [item[0] for item in data["frame"]]

# Convert centroids to a NumPy array for efficient calculations
centroids = np.array(centroids)  # Shape (N, 2)
x = centroids[:, 0]  # X-coordinates
y = centroids[:, 1]  # Y-coordinates

# =============================================================================
# 3) CALCULATE DISPLACEMENTS AND VELOCITIES
# =============================================================================
# Calculate differences (displacements) between consecutive frames
dx = np.diff(x)
dy = np.diff(y)

# Calculate velocities in x and y directions
fps = 200.0  # Frames per second
dt = 1 / fps  # Time interval between frames
vx = dx / dt
vy = dy / dt

# Associate velocities with frames (starting from the second frame)
frames_v = frames[1:]

# =============================================================================
# 4) SMOOTH SIGNALS USING A MOVING AVERAGE FILTER
# =============================================================================
def smooth_signal(signal, window_size):
    """
    Smooth a signal using a moving average filter.
    Args:
        signal (np.ndarray): Input signal to be smoothed.
        window_size (int): Size of the moving average window.
    Returns:
        np.ndarray: Smoothed signal.
    """
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(signal, window, mode='same')

# Define the window size for smoothing
window_size = 5

# Apply smoothing to the velocity signals
vx_smooth = smooth_signal(vx, window_size)
vy_smooth = smooth_signal(vy, window_size)

# =============================================================================
# 5) PLOT VELOCITIES (ORIGINAL VS SMOOTHED)
# =============================================================================
# Create subplots for the X and Y velocities
fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Plot velocity in X
axs[0].plot(frames_v, vx, marker='.', linestyle=':', color='lightblue', label="Original $v_x$")
axs[0].plot(frames_v, vx_smooth, marker='o', linestyle='-', color='blue', label="Smoothed $v_x$")
axs[0].set_ylabel("Velocity in X (units/s)")
axs[0].set_title("Velocity in X")
axs[0].legend()
axs[0].grid(True)

# Plot velocity in Y
axs[1].plot(frames_v, vy, marker='.', linestyle=':', color='salmon', label="Original $v_y$")
axs[1].plot(frames_v, vy_smooth, marker='o', linestyle='-', color='red', label="Smoothed $v_y$")
axs[1].set_xlabel("Frame")
axs[1].set_ylabel("Velocity in Y (units/s)")
axs[1].set_title("Velocity in Y")
axs[1].legend()
axs[1].grid(True)

# Add a main title and adjust layout
#plt.suptitle("Velocities (Original vs. Smoothed) from Centroids (fps = 200)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Show the plot
plt.savefig("Graphs/convolutional-velocities.png")