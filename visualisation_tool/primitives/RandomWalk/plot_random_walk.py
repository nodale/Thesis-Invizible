"""Visualise RandomWalk: 5 episodes chained in continuous time, each with a new random velocity."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

N_EPISODES  = 5
VEL_SCALE   = 0.2
STEPS_PER   = 100        # steps per episode (τ ∈ [0, 1] within each)


def _grid(ax):
    for which, a, lw in [("major", 0.35, 0.7), ("minor", 0.15, 0.4)]:
        ax.grid(which=which, alpha=a, linewidth=lw)
    ax.xaxis.set_major_locator(MaxNLocator(8)); ax.xaxis.set_minor_locator(MaxNLocator(32))
    ax.yaxis.set_major_locator(MaxNLocator(8)); ax.yaxis.set_minor_locator(MaxNLocator(32))


def main():
    rng   = np.random.default_rng(3)
    tau   = np.linspace(0, 1, STEPS_PER)

    fig, (ax_xy, ax_z) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"wspace": 0.3})
    fig.suptitle("RandomWalk — 5 consecutive episodes", fontsize=11)

    colors = plt.cm.tab10(np.linspace(0, 0.5, N_EPISODES))
    pos    = np.zeros(3)          # carried forward between episodes
    t_global = 0.0

    for ep in range(N_EPISODES):
        vel  = rng.standard_normal(3) * VEL_SCALE
        traj = pos + np.outer(tau, vel)         # (STEPS_PER, 3)
        t    = t_global + tau

        label = f"walk number {ep + 1}"
        ax_xy.plot(*traj[:, :2].T, lw=1.6, color=colors[ep], label=label)
        ax_z.plot(t, traj[:, 2],   lw=1.6, color=colors[ep], label=label)

        # mark episode boundary
        ax_xy.plot(*traj[0,  :2], "o", ms=5, color=colors[ep], zorder=5)
        ax_z.axvline(t_global, lw=0.6, ls=":", color="gray", alpha=0.6)

        pos      = traj[-1]
        t_global += 1.0

    ax_xy.plot(*pos[:2], "x", ms=7, color="black", zorder=5, label="final")
    ax_xy.set_xlabel("x (m)"); ax_xy.set_ylabel("y (m)")
    ax_xy.set_title("XY plane"); ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.legend(fontsize=8); _grid(ax_xy)

    ax_z.set_xlabel("time (episodes)"); ax_z.set_ylabel("z (m)")
    ax_z.set_title("Z over time"); ax_z.legend(fontsize=8); _grid(ax_z)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("random_walk.png", dpi=150)
    print("Saved random_walk.png")


if __name__ == "__main__":
    main()
