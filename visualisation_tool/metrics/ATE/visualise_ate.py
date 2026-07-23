"""
Absolute Trajectory Error (ATE) visualisation.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

def bezier(p0, p1, p2, p3, n=100):
    t = np.linspace(0, 1, n)[:, None]
    return (1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3

# Ground truth bezier
gt = bezier([0,0], [1,3], [3,3], [4,0])

# Estimated: add drift + noise
noise = rng.normal(0, 0.08, gt.shape)
drift = np.column_stack([np.linspace(0, 0.3, 100), np.linspace(0, -0.2, 100)])
est = gt + drift + noise
est -= est.mean(axis=0) - gt.mean(axis=0)  # centroid alignment

errors = np.linalg.norm(gt - est, axis=1)
ate = np.sqrt(np.mean(errors ** 2))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle(f"Absolute Trajectory Error  —  ATE = {ate:.3f} m", fontsize=12)

ax = axes[0]
ax.plot(gt[:, 0],  gt[:, 1],  "g-",  lw=2,   label="Ground truth")
ax.plot(est[:, 0], est[:, 1], "b--", lw=1.5, label="Estimated (aligned)")
ax.set_title("Trajectory")
ax.axis("equal")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(errors, color="steelblue", lw=1.5)
ax.axhline(ate, color="red", ls="--", label=f"RMSE = {ate:.3f}")
ax.fill_between(range(len(errors)), errors, alpha=0.2, color="steelblue")
ax.set_xlabel("Frame")
ax.set_ylabel("Error (m)")
ax.set_title("Per-frame error")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ate_visualisation.png", dpi=150)
plt.show()
