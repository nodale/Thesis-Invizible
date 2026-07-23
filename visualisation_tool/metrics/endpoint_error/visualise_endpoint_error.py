"""
Endpoint Error visualisation.
Euclidean distance between predicted and ground-truth final position.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

def bezier(p0, p1, p2, p3, n=100):
    t = np.linspace(0, 1, n)[:, None]
    return (1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3

gt   = bezier([0,0], [1,2], [2,2], [3,1])
pred = bezier([0,0], [1.3,1.8], [2.2,2.4], [3,1] + rng.normal(0, 0.4, 2))

gt_end   = gt[-1]
pred_end = pred[-1]
ee = np.linalg.norm(pred_end - gt_end)

plt.figure(figsize=(6, 5))
plt.plot(gt[:, 0],   gt[:, 1],   "g-",  lw=2,   label="Ground truth")
plt.plot(pred[:, 0], pred[:, 1], "b--", lw=1.5, label="Predicted")
plt.plot([gt_end[0], pred_end[0]], [gt_end[1], pred_end[1]], "r-", lw=2,
         label=f"Endpoint error = {ee:.3f} m")
plt.scatter(*gt_end,   color="green", s=80, zorder=5)
plt.scatter(*pred_end, color="blue",  s=80, zorder=5)
plt.title("Endpoint Error")
plt.legend()
plt.axis("equal")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("endpoint_error_visualisation.png", dpi=150)
plt.show()
