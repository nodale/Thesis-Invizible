"""Visualise M episodes of CircularOrbit / RandomSphereOffset / RandomWalk sampled with equal weight.

3-D trajectory coloured by time (early = purple, late = yellow).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

M           = 15
STEPS_PER   = 120
VEL_SCALE   = 0.2
MIN_RADIUS  = 0.05
MAX_RADIUS  = 0.3
MIN_LAPS    = 0.2
MAX_LAPS    = 1.0
CMAP        = "plasma"


def _circular_orbit(start, rng, tau):
    r    = rng.uniform(MIN_RADIUS, MAX_RADIUS)
    laps = rng.uniform(MIN_LAPS, MAX_LAPS)
    sign = rng.choice([-1, 1])
    phi  = laps * 2 * np.pi * sign * tau
    traj = np.empty((len(tau), 3))
    traj[:, 0] = start[0] + r * np.cos(phi)
    traj[:, 1] = start[1] + r * np.sin(phi)
    traj[:, 2] = start[2]
    return traj


def _random_sphere_offset(start, rng, tau):
    direction  = rng.standard_normal(3)
    direction /= np.linalg.norm(direction)
    radius     = rng.uniform(MIN_RADIUS, MAX_RADIUS)
    target     = start + direction * radius
    return np.outer(np.ones(len(tau)), target)


def _random_walk(start, rng, tau):
    vel = rng.standard_normal(3) * VEL_SCALE
    return start + np.outer(tau, vel)


ACTIONS = [_circular_orbit, _random_sphere_offset, _random_walk]


def _colored_line3d(ax, pts, t_norm, cmap, lw=1.8, alpha=0.9):
    """Draw a 3-D polyline with per-segment colour from a normalised time array."""
    segs   = np.stack([pts[:-1], pts[1:]], axis=1)          # (N-1, 2, 3)
    colors = plt.get_cmap(cmap)((t_norm[:-1] + t_norm[1:]) / 2)
    lc     = Line3DCollection(segs, colors=colors, linewidth=lw, alpha=alpha)
    ax.add_collection(lc)


def main():
    rng = np.random.default_rng(8)
    tau = np.linspace(0, 1, STEPS_PER)

    # ── collect full trajectory ────────────────────────────────────────────────
    all_pts  = []
    all_t    = []
    connectors = []          # (prev_end, next_start, t_prev, t_next) for gap segments

    pos      = np.zeros(3)
    t_global = 0.0
    prev_end = None
    prev_t   = None

    for ep in range(M):
        traj = ACTIONS[rng.integers(0, 3)](pos, rng, tau)
        t    = t_global + tau

        if prev_end is not None:
            connectors.append((prev_end.copy(), traj[0].copy(), prev_t, t[0]))

        all_pts.append(traj)
        all_t.append(t)

        prev_end  = traj[-1]
        prev_t    = t[-1]
        pos       = traj[-1]
        t_global += 1.0

    pts   = np.concatenate(all_pts, axis=0)
    t_all = np.concatenate(all_t)
    t_norm = (t_all - t_all[0]) / (t_all[-1] - t_all[0])

    # ── plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection="3d")
    fig.suptitle(f"Combined actions — {M} episodes, equal weighting", fontsize=11)

    _colored_line3d(ax, pts, t_norm, CMAP)

    # connectors (thin grey dashes)
    for p0, p1, t0, t1 in connectors:
        seg = np.array([p0, p1])
        ax.plot(*seg.T, lw=0.9, ls="--", color="gray", alpha=0.5)

    # start / end markers
    ax.scatter(*pts[0],  s=50, color=plt.get_cmap(CMAP)(0.0), zorder=5, label="start")
    ax.scatter(*pts[-1], s=50, color=plt.get_cmap(CMAP)(1.0), zorder=5, label="end")

    # colourbar
    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=plt.Normalize(t_all[0], t_all[-1]))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=0.1, shrink=0.6, label="time (episodes)")

    pad = 0.02
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    ax.set_zlim(pts[:, 2].max() + pad, pts[:, 2].min() - pad)
    ax.set_box_aspect([
        np.ptp(pts[:, 0]),
        np.ptp(pts[:, 1]),
        np.ptp(pts[:, 2]*3),
    ])
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.view_init(elev=15, azim=-45)
    ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("combined.png", dpi=150)
    print("Saved combined.png")


if __name__ == "__main__":
    main()
