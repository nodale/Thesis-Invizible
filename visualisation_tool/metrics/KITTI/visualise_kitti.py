"""
KITTI Odometry Translation Drift visualisation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator

rng = np.random.default_rng(0)


# KITTI evaluation segment lengths
lengths = np.array(
    [100, 200, 300, 400, 500, 600, 700, 800]
)

# Simulated KITTI translational drift (%)
t_err = np.clip(
    1.5 + rng.normal(0, 0.3, len(lengths)),
    0.1,
    None,
)

mean_drift = np.mean(t_err)


fig, ax = plt.subplots(
    figsize=(10, 4),
    dpi=50,
)


# Translation drift curve
ax.plot(
    lengths,
    t_err,
    "-o",
    color="red",
    lw=1.4,
    ms=5,
    mfc="red",
    mec="red",
    zorder=3,
)


# Average drift line
ax.axhline(
    mean_drift,
    color="black",
    ls=(0, (4, 3)),
    lw=1.2,
)


# Annotation
ax.text(
    lengths[-2],
    mean_drift + 0.05,
    f"Mean drift = {mean_drift:.2f} %",
    fontsize=11,
    color="black",
)


# Limits with whitespace
ax.set_xlim(
    lengths[0] - 30,
    lengths[-1] + 30,
)

ax.set_ylim(
    0,
    max(t_err) * 1.25,
)


# Clean style matching ATE figure
#ax.set_title("KITTI translation drift")
ax.set_xlabel("Segment length (m)")
ax.set_ylabel("Translation error (%)")
#ax.grid(True, alpha=0.3)
#ax.set_axis_off()
ax.xaxis.set_major_locator(MaxNLocator(8))
ax.yaxis.set_major_locator(MaxNLocator(8))
ax.xaxis.set_minor_locator(MaxNLocator(32))
ax.yaxis.set_minor_locator(MaxNLocator(32))
ax.grid(which="major", alpha=0.35, linewidth=0.7)
ax.grid(which="minor", alpha=0.15, linewidth=0.4)


plt.tight_layout()
plt.savefig(
    "kitti_translation_drift_visualisation.png",
    dpi=400,
)

#plt.show()
