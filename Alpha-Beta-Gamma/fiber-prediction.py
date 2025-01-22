import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

# =============================================================================
# 1) GLOBAL STYLE SETTINGS
# =============================================================================
# You can adjust these RC parameters to match your paper/journal requirements.
# e.g. Use Times, a certain font size, etc.
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

# Alternatively, you can pick a style:
# plt.style.use('seaborn-whitegrid')

# =============================================================================
# 2) LOAD DATA (Modificación: Limitar Frames)
# =============================================================================
json_path = 'Alpha-Beta-Gamma/fibra_21.json'
max_frames_to_process = 100  # Límite de frames a procesar

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Limitar los datos al número máximo de frames
centroids = data['centroide'][:max_frames_to_process]
angles_list = data['angulo'][:max_frames_to_process]
frames = data['frame'][:max_frames_to_process]

fps = 200
delta_t = 1.0 / fps
num_frames = len(centroids)

# =============================================================================
# 3) REAL DISTANCE, VELOCITY, ANGLE, ANGULAR VELOCITY
# =============================================================================

# 3.1) Distance from bottom-left (0,0)
real_distance = np.zeros(num_frames)
for i in range(num_frames):
    x, y = centroids[i]
    real_distance[i] = np.sqrt(x**2 + y**2)

# 3.2) Velocity magnitude
real_velocity = []
for i in range(num_frames - 1):
    dx = centroids[i+1][0] - centroids[i][0]
    dy = centroids[i+1][1] - centroids[i][1]
    vx = dx / delta_t
    vy = dy / delta_t
    real_velocity.append([vx, vy])
real_velocity = np.array(real_velocity)
real_vel_mag = np.sqrt(real_velocity[:, 0]**2 + real_velocity[:, 1]**2)
frames_vel = frames[1:]

# 3.3) Angles (degrees)
real_angle = np.array([item[0] for item in angles_list])

# 3.4) Angular velocity
real_ang_velocity = []
for i in range(num_frames - 1):
    da = real_angle[i+1] - real_angle[i]
    w = da / delta_t
    real_ang_velocity.append(w)
real_ang_velocity = np.array(real_ang_velocity)
frames_ang_vel = frames[1:]

# =============================================================================
# 4) ALPHA-BETA-GAMMA FILTER
# =============================================================================
def alpha_beta_gamma_filter(centroids, angles_deg, alpha, beta, gamma, dt):
    """
    Applies Alpha-Beta-Gamma filtering to (x, y) position and angle (in degrees).
    Returns:
        pred_positions  (N, 2): (x, y) for each frame
        pred_velocities (N, 2): (vx, vy) for each frame
        pred_angles     (N, ):  angle in degrees for each frame
        pred_angvel     (N, ):  angular velocity (deg/s) for each frame
    """
    n = len(centroids)

    pred_positions = np.zeros((n, 2), dtype=float)
    pred_velocities = np.zeros((n, 2), dtype=float)
    pred_angles = np.zeros(n, dtype=float)
    pred_angvel = np.zeros(n, dtype=float)

    # Initial states
    x_est = centroids[0][0]
    y_est = centroids[0][1]
    vx_est = 0.0
    vy_est = 0.0
    ax_est = 0.0
    ay_est = 0.0

    ang_est = angles_deg[0]
    w_est = 0.0
    alpha_est = 0.0  # 'angular acceleration' for the filter

    # Store the initial guess
    pred_positions[0] = [x_est, y_est]
    pred_velocities[0] = [vx_est, vy_est]
    pred_angles[0] = ang_est
    pred_angvel[0] = w_est

    for i in range(1, n):
        # Current measurements
        z_x = centroids[i][0]
        z_y = centroids[i][1]
        z_ang = angles_deg[i]

        # --- Correction (Alpha, Beta, Gamma) for position ---
        x_est = x_est + alpha * (z_x - x_est)
        y_est = y_est + alpha * (z_y - y_est)
        vx_est = vx_est + beta * ((z_x - x_est) / dt)
        vy_est = vy_est + beta * ((z_y - y_est) / dt)
        ax_est = ax_est + gamma * ((z_x - x_est) / (2 * dt**2))
        ay_est = ay_est + gamma * ((z_y - y_est) / (2 * dt**2))

        # --- Correction for angle ---
        ang_est = ang_est + alpha * (z_ang - ang_est)
        w_est = w_est + beta * ((z_ang - ang_est) / dt)
        alpha_est = alpha_est + gamma * ((z_ang - ang_est) / (2 * dt**2))

        # --- Prediction step (Alpha-Beta-Gamma logic) ---
        x_pred = x_est + vx_est * dt + 0.5 * ax_est * (dt**2)
        y_pred = y_est + vy_est * dt + 0.5 * ay_est * (dt**2)
        vx_pred = vx_est + ax_est * dt
        vy_pred = vy_est + ay_est * dt

        ang_pred = ang_est + w_est * dt + 0.5 * alpha_est * (dt**2)
        w_pred = w_est + alpha_est * dt

        # Update states
        x_est, y_est = x_pred, y_pred
        vx_est, vy_est = vx_pred, vy_pred
        ang_est, w_est = ang_pred, w_pred

        # Store predictions
        pred_positions[i] = [x_est, y_est]
        pred_velocities[i] = [vx_est, vy_est]
        pred_angles[i] = ang_est
        pred_angvel[i] = w_est

    return pred_positions, pred_velocities, pred_angles, pred_angvel

# =============================================================================
# 5) PARAMETER COMBINATIONS & COMPUTE PREDICTIONS
# =============================================================================
param_combos = [
    (0.2, 0.2, 0.02),
    (0.5, 0.5, 0.05),
    (0.8, 0.8, 0.08)
]

pred_results = {}
for (alpha, beta, gamma) in param_combos:
    ppos, pvel, pang, pangvel = alpha_beta_gamma_filter(
        centroids, real_angle, alpha, beta, gamma, delta_t
    )

    dist_pred = np.sqrt(ppos[:, 0]**2 + ppos[:, 1]**2)
    vel_mag_all = np.sqrt(pvel[:, 0]**2 + pvel[:, 1]**2)
    vel_mag_aligned = vel_mag_all[1:]

    pred_results[(alpha, beta, gamma)] = {
        'distance': dist_pred,
        'vel_mag': vel_mag_aligned,
        'angle': pang,
        'angvel': pangvel[1:]
    }

# =============================================================================
# GRÁFICAS (SE LIMITAN A LOS FRAMES)
# =============================================================================
# Las gráficas permanecen igual, usando `frames`, `real_distance`, y `pred_results`.


# =============================================================================
# 6) COMPUTE PREDICTIONS
# =============================================================================
for (alpha, beta, gamma) in param_combos:
    ppos, pvel, pang, pangvel = alpha_beta_gamma_filter(
        centroids, real_angle, alpha, beta, gamma, delta_t
    )

    # Distance from (0,0)
    dist_pred = np.sqrt(ppos[:, 0]**2 + ppos[:, 1]**2)
    # Velocity magnitude (align frames[1..])
    vel_mag_all = np.sqrt(pvel[:, 0]**2 + pvel[:, 1]**2)
    vel_mag_aligned = vel_mag_all[1:]
    # Angle
    ang_all = pang
    # Angular velocity (align frames[1..])
    angvel_all = pangvel[1:]

    pred_results[(alpha, beta, gamma)] = {
        'distance': dist_pred,
        'vel_mag': vel_mag_aligned,
        'angle': ang_all,
        'angvel': angvel_all
    }

# =============================================================================
# 7) FIGURE 1 -> DISTANCE & VELOCITY
# =============================================================================
fig1, axs1 = plt.subplots(2, 1, figsize=(7, 8), sharex=False)

ax_dist = axs1[0]
ax_dist.set_title("Distance from (0,0)", pad=10)
ax_dist.plot(frames, real_distance, color='black', label="Real Distance")
for (alpha, beta, gamma), vals in pred_results.items():
    ax_dist.plot(
        frames, vals['distance'],
        linestyle='--',
        label=f"Pred (α={alpha}, β={beta}, γ={gamma})"
    )
ax_dist.set_ylabel("Distance [pixels]")
ax_dist.grid(True)
ax_dist.legend(loc='best')
# Remove top/right spines for a cleaner look
ax_dist.spines['top'].set_visible(False)
ax_dist.spines['right'].set_visible(False)

ax_vel = axs1[1]
ax_vel.set_title("Velocity Magnitude", pad=10)
ax_vel.plot(frames_vel, real_vel_mag, color='black', label="Real Velocity")
for (alpha, beta, gamma), vals in pred_results.items():
    ax_vel.plot(
        frames_vel, vals['vel_mag'],
        linestyle='--',
        label=f"Pred (α={alpha}, β={beta}, γ={gamma})"
    )
ax_vel.set_xlabel("Frame")
ax_vel.set_ylabel("Velocity [px/s]")
ax_vel.grid(True)
ax_vel.legend(loc='best')
ax_vel.spines['top'].set_visible(False)
ax_vel.spines['right'].set_visible(False)

plt.tight_layout()
os.makedirs("Alpha-Beta-Gamma/Graph", exist_ok=True)
out_fig1 = "Alpha-Beta-Gamma/Graph/distance_velocity_paper.png"
plt.savefig(out_fig1, dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved Figure 1 -> {out_fig1}")

# =============================================================================
# 8) FIGURE 2 -> ANGLE & ANGULAR VELOCITY
# =============================================================================
fig2, axs2 = plt.subplots(2, 1, figsize=(7, 8), sharex=False)

ax_angle = axs2[0]
ax_angle.set_title("Angle (Degrees)", pad=10)
ax_angle.plot(frames, real_angle, color='black', label="Real Angle")
for (alpha, beta, gamma), vals in pred_results.items():
    ax_angle.plot(
        frames, vals['angle'],
        linestyle='--',
        label=f"Pred (α={alpha}, β={beta}, γ={gamma})"
    )
ax_angle.set_ylabel("Angle [deg]")
ax_angle.grid(True)
ax_angle.legend(loc='best')
ax_angle.spines['top'].set_visible(False)
ax_angle.spines['right'].set_visible(False)

ax_angvel = axs2[1]
ax_angvel.set_title("Angular Velocity (Degrees/s)", pad=10)
ax_angvel.plot(frames_ang_vel, real_ang_velocity, color='black', label="Real Ang Vel")
for (alpha, beta, gamma), vals in pred_results.items():
    ax_angvel.plot(
        frames_ang_vel, vals['angvel'],
        linestyle='--',
        label=f"Pred (α={alpha}, β={beta}, γ={gamma})"
    )
ax_angvel.set_xlabel("Frame")
ax_angvel.set_ylabel("Ang. Velocity [deg/s]")
ax_angvel.grid(True)
ax_angvel.legend(loc='best')
ax_angvel.spines['top'].set_visible(False)
ax_angvel.spines['right'].set_visible(False)

plt.tight_layout()
out_fig2 = "Alpha-Beta-Gamma/Graph/angle_angvel_paper.png"
plt.savefig(out_fig2, dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved Figure 2 -> {out_fig2}")
