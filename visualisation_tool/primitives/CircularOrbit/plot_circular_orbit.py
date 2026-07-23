"""Visualise CircularOrbit setpoints: one example with annotated parameters."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

MIN_RADIUS  = 0.05
MAX_RADIUS  = 0.3
MIN_LAPS    = 0.2
MAX_LAPS    = 1.0
N_STEPS     = 300


def _grid(ax):
    for which, a, lw in [("major", 0.35, 0.7), ("minor", 0.15, 0.4)]:
        ax.grid(which=which, alpha=a, linewidth=lw)
    ax.xaxis.set_major_locator(MaxNLocator(8)); ax.xaxis.set_minor_locator(MaxNLocator(32))
    ax.yaxis.set_major_locator(MaxNLocator(8)); ax.yaxis.set_minor_locator(MaxNLocator(32))


def circular_orbit(start_pos, radius, total_angle, tau):
    phi = total_angle * tau
    x = start_pos[0] + radius * np.cos(phi)
    y = start_pos[1] + radius * np.sin(phi)
    return np.column_stack([x, y])


def main():
    rng   = np.random.default_rng(42)
    tau   = np.linspace(0, 1, N_STEPS)
    start = np.zeros(3)
    r     = rng.uniform(MIN_RADIUS, MAX_RADIUS)
    laps  = rng.uniform(MIN_LAPS, MAX_LAPS)
    sign  = rng.choice([-1, 1])
    total = laps * 2 * np.pi * sign
    traj  = circular_orbit(start, r, total, tau)

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.suptitle("CircularOrbit — setpoint trajectory", fontsize=11)

    ax.plot(*traj.T, lw=1.8, color="tab:blue", label="setpoint")
    ax.plot(*traj[0],  "o", ms=6, color="black",   zorder=5, label="start (τ=0)")
    ax.plot(*traj[-1], "x", ms=8, color="tab:red",  zorder=5, label="end  (τ=1)")

    # radius annotation
    cx, cy = start[0], start[1]
    mid_phi = total * 0.5
    rx, ry  = cx + r * np.cos(mid_phi), cy + r * np.sin(mid_phi)
    ax.annotate("", xy=(rx, ry), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="<->", lw=1.2, color="dimgray"))
    ax.text((cx + rx) / 2 + 0.005, (cy + ry) / 2 + 0.005,
            f"r = {r:.3f} m", fontsize=8, color="dimgray")

    direction = "CCW" if sign > 0 else "CW"
    ax.set_title(f"r = {r:.3f} m  |  laps = {laps:.2f}  |  {direction}", fontsize=9)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=8)
    _grid(ax)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("circular_orbit.png", dpi=150)
    print("Saved circular_orbit.png")


if __name__ == "__main__":
    main()
