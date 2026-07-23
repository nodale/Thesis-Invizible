"""
KITTI Odometry Metrics visualisation.
t_err: avg translational error (%) over sub-sequences
r_err: avg rotational error (deg/m) over sub-sequences
Ground truth = 0 drift (perfect odometry baseline).
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

lengths = np.array([100, 200, 300, 400, 500, 600, 700, 800])

t_err = np.clip(1.5 + rng.normal(0, 0.3, len(lengths)), 0.1, None)
r_err = np.clip(0.005 + rng.normal(0, 0.003, len(lengths)), 0.001, None)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("KITTI Odometry Metrics")

axes[0].plot(lengths, np.zeros_like(lengths), "g-", lw=1.5, label="Ground truth (0%)")
axes[0].plot(lengths, t_err, "o-", color="steelblue", label=f"Estimated (avg {t_err.mean():.2f}%)")
axes[0].set_xlabel("Sub-sequence length (m)")
axes[0].set_ylabel("Translational error (%)")
axes[0].set_title("t_err")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(lengths, np.zeros_like(lengths), "g-", lw=1.5, label="Ground truth (0)")
axes[1].plot(lengths, r_err * 1000, "o-", color="tomato",
             label=f"Estimated (avg {r_err.mean()*1000:.3f} ×10⁻³ deg/m)")
axes[1].set_xlabel("Sub-sequence length (m)")
axes[1].set_ylabel("Rotational error (×10⁻³ deg/m)")
axes[1].set_title("r_err")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("kitti_visualisation.png", dpi=150)
plt.show()
