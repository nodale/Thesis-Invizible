"""
Endpoint Error visualisation.
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

def bezier(p0, p1, p2, p3, n=100):
    t = np.linspace(0, 1, n)[:, None]
    return (1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t**2*p2 + t**3*p3


gt = bezier([0,0], [1,2], [2,2], [3,1])

pred = bezier(
    [0,0],
    [1.3,1.8],
    [2.2,2.4],
    [3,1] + rng.normal(0, 2.0, 2)
)

gt_end = gt[-1]
pred_end = pred[-1]

ee = np.linalg.norm(pred_end - gt_end)


fig, ax = plt.subplots(
    figsize=(12, 4),
    dpi=50,
)


# Endpoint correspondence line
ax.plot(
    [gt_end[0], pred_end[0]],
    [gt_end[1], pred_end[1]],
    color="0.45",
    ls=(0, (4, 4)),
    lw=1.0,
    alpha=0.8,
    zorder=1,
)


# Ground truth trajectory
ax.plot(
    gt[:, 0],
    gt[:, 1],
    color="black",
    lw=1.4,
    zorder=2,
)

# Estimated trajectory
ax.plot(
    pred[:, 0],
    pred[:, 1],
    color="red",
    lw=1.4,
    zorder=2,
)


# Endpoint markers
ax.plot(
    gt_end[0],
    gt_end[1],
    "o",
    color="black",
    ms=5,
    zorder=3,
)

ax.plot(
    pred_end[0],
    pred_end[1],
    "o",
    color="red",
    ms=5,
    zorder=3,
)


# Add whitespace around trajectory
pad = 0.3
xmin = min(gt[:,0].min(), pred[:,0].min(), gt_end[0], pred_end[0]) - pad
xmax = max(gt[:,0].max(), pred[:,0].max(), gt_end[0], pred_end[0]) + pad
ymin = min(gt[:,1].min(), pred[:,1].min(), gt_end[1], pred_end[1]) - pad
ymax = max(gt[:,1].max(), pred[:,1].max(), gt_end[1], pred_end[1]) + pad

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)


# Error annotation
ax.text(
    0.65 * (xmax - xmin) + xmin,
    ymin + 0.08 * (ymax - ymin),
    f"Endpoint error = {ee:.3f} m",
    fontsize=11,
    color="black",
)


# Same clean style as ATE figure
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
#ax.axis("equal")
#ax.set_title("Endpoint Error")
#ax.legend(fontsize=8)
#ax.grid(True, alpha=0.3)
#ax.set_axis_off()


plt.tight_layout()
plt.savefig(
    "endpoint_error_visualisation.png",
    dpi=200,
)
plt.show()
