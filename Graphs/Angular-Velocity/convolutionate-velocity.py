import json
import numpy as np

# =============================================================================
# 1) LOAD DATA FROM JSON
# =============================================================================
# Specify the JSON file path
json_file = "Graphs/Angular-Velocity/fibras_filtradas.json"  # Replace with your actual JSON file path

# Load the JSON data
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# =============================================================================
# 2) DEFINE HELPER FUNCTIONS
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

def angular_difference(angle1, angle2):
    """
    Compute the smallest difference between two angles, ensuring it falls in [-180, 180].
    Args:
        angle1 (float): First angle in degrees.
        angle2 (float): Second angle in degrees.
    Returns:
        float: Angular difference in degrees.
    """
    diff = angle1 - angle2
    while diff > 180:
        diff -= 360
    while diff <= -180:
        diff += 360
    return diff

# =============================================================================
# 3) PROCESS EACH FIBER
# =============================================================================
# Define the window size for convolution
window_size = 5

# Iterate over each fiber in the JSON
for fiber_id, fiber_data in data.items():
    if fiber_id not in ["ruta", "fibras_por_frame"]:  # Skip metadata keys
        # Extract angles
        angles = np.array([item[0] for item in fiber_data["angulo"]])  # Extract the angle list

        # Calculate angular differences (angular velocity)
        fps = 200.0  # Frames per second
        dt = 1 / fps  # Time interval between frames

        angular_velocities = [
            angular_difference(angles[i + 1], angles[i]) / dt
            for i in range(len(angles) - 1)
        ]
        angular_velocities = np.array(angular_velocities)

        # Smooth angular velocities using the moving average filter
        angular_velocities_smooth = smooth_signal(angular_velocities, window_size)

        # Add smoothed angular velocities back to the fiber data
        # Note: We pad the smoothed array to match the number of frames
        fiber_data["velocidad_angular_convolucionada"] = angular_velocities_smooth.tolist()

# =============================================================================
# 4) SAVE THE UPDATED JSON
# =============================================================================
# Specify the output JSON file path
output_file = "Graphs/Angular-Velocity/fiber_convolutionated.json"  # Replace with your desired output file name

# Save the updated data to a new JSON file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

print(f"Updated JSON with angular velocities saved to: {output_file}")
