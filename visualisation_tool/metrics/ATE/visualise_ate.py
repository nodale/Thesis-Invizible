"""
Absolute Trajectory Error (ATE) visualisation.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(400)

def bezier(p0, p1, p2, p3, n=10):
    t = np.linspace(0, 1, n)[:, None]
    return (1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3

gt = bezier([0,0], [1,3], [3,-3], [4,0])
noise = rng.normal(0, 0.05, gt.shape)
tangent = np.gradient(gt, axis=0)
tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])

drift_mag = np.linspace(0, 0.6, len(gt))
drift = normal * drift_mag[:, None]

est = gt + drift + noise

errors = np.linalg.norm(gt - est, axis=1)
ate = np.sqrt(np.mean(errors ** 2))

fig, axes = plt.subplots(
    1, 2,
    figsize=(20, 4),
    dpi=50,
    gridspec_kw={"wspace": 0.15},
)
#fig.suptitle(f"Absolute Trajectory Error  —  ATE = {ate:.3f} m", fontsize=11)

ax = axes[0]

# Dashed correspondence lines
for p_gt, p_est in zip(gt, est):
    ax.plot(
        [p_gt[0], p_est[0]],
        [p_gt[1], p_est[1]],
        color="0.45",
        ls=(0, (4, 4)),   # nicer dashes
        lw=1.0,
        alpha=0.8,
        zorder=1,
    )

# Ground truth
ax.plot(
    gt[:, 0], gt[:, 1],
    "-o",
    color="black",
    lw=1.4,
    ms=5,
    mfc="black",
    mec="black",
    zorder=3,
)

# Estimated
ax.plot(
    est[:, 0], est[:, 1],
    "-o",
    color="red",
    lw=1.4,
    ms=5,
    mfc="red",
    mec="red",
    zorder=3,
)

# Add some whitespace around the trajectory
pad = 0.3
xmin = min(gt[:,0].min(), est[:,0].min()) - pad
xmax = max(gt[:,0].max(), est[:,0].max()) + pad
ymin = min(gt[:,1].min(), est[:,1].min()) - pad
ymax = max(gt[:,1].max(), est[:,1].max()) + pad
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
#ax.axis("equal")
#ax.set_title("Trajectory")
#ax.legend(fontsize=8)
#ax.grid(True, alpha=0.3)
#ax.set_axis_off()

ax = axes[1]
frames = np.arange(len(errors))
ax.plot(
    frames,
    errors,
    "-o",
    color="red",
    lw=1.4,
    ms=5,
)
ax.axhline(
    ate,
    color="black",
    ls=(0, (4, 3)),
    lw=1.2,
)
ax.text(
    len(errors) * 0.67,
    ate + 0.01,
    f"RMSE = {ate:.3f}",
    fontsize=11,
    color="black",
)
ax.set_xlim(-0.5, len(errors)-0.5)
ax.set_ylim(0, max(errors) * 1.15)
ax.set_xlabel("trajectory Index")
ax.set_ylabel("ATE per frame (m)")
#ax.set_title("Trajectory")
#ax.legend(fontsize=8)
#ax.grid(True, alpha=0.3)
#ax.set_axis_off()

plt.tight_layout()
plt.savefig("ate_visualisation.png", dpi=200)
#plt.show()
