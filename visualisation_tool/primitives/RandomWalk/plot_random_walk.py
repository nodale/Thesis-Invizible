"""Visualise RandomWalk: 5 chained episodes in 3D, coloured by time."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

N_EPISODES  = 5
VEL_SCALE   = 0.2
STEPS_PER   = 100
CMAP        = "plasma"


def _colored_line3d(ax, pts, t_norm, cmap, lw=1.8, alpha=0.9):
    segs   = np.stack([pts[:-1], pts[1:]], axis=1)
    colors = plt.get_cmap(cmap)((t_norm[:-1] + t_norm[1:]) / 2)
    lc     = Line3DCollection(segs, colors=colors, linewidth=lw, alpha=alpha)
    ax.add_collection(lc)


def main():
    rng      = np.random.default_rng(3)
    tau      = np.linspace(0, 1, STEPS_PER)
    pos      = np.zeros(3)
    t_global = 0.0

    all_pts = []
    all_t   = []
    bounds  = []   # episode start indices for boundary markers

    for ep in range(N_EPISODES):
        vel  = rng.standard_normal(3) * VEL_SCALE
        traj = pos + np.outer(tau, vel)
        bounds.append(len(np.concatenate(all_pts)) if all_pts else 0)
        all_pts.append(traj)
        all_t.append(t_global + tau)
        pos       = traj[-1]
        t_global += 1.0

    pts    = np.concatenate(all_pts, axis=0)
    t_all  = np.concatenate(all_t)
    t_norm = (t_all - t_all[0]) / (t_all[-1] - t_all[0])

    fig = plt.figure(figsize=(8, 7))
    ax  = fig.add_subplot(111, projection="3d")
    fig.suptitle(f"RandomWalk — {N_EPISODES} consecutive episodes", fontsize=11)

    _colored_line3d(ax, pts, t_norm, CMAP)

    # episode boundary dots
    for i, idx in enumerate(bounds):
        c = plt.get_cmap(CMAP)(t_norm[idx])
        ax.scatter(*pts[idx], s=30, color=c, zorder=5)

    ax.scatter(*pts[0],  s=60, color=plt.get_cmap(CMAP)(0.0), zorder=6, label="start")
    ax.scatter(*pts[-1], s=60, color=plt.get_cmap(CMAP)(1.0), zorder=6, label="end")

    pad = 0.02
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    ax.set_zlim(pts[:, 2].max() + pad, pts[:, 2].min() - pad)   # -z up
    ax.set_box_aspect([np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), np.ptp(pts[:, 2])])

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(t_all[0], t_all[-1]))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.6, label="time (episodes)")

    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("random_walk.png", dpi=150)
    print("Saved random_walk.png")


if __name__ == "__main__":
    main()
