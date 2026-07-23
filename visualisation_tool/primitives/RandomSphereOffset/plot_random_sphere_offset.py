"""Visualise RandomSphereOffset: one sampled setpoint on a semi-transparent sphere shell."""

import numpy as np
import matplotlib.pyplot as plt

MIN_RADIUS  = 0.05
MAX_RADIUS  = 0.2


def sphere_mesh(radius, n=40):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi,     n)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(n), np.cos(v))
    return x, y, z


def main():
    rng    = np.random.default_rng(8)
    start  = np.zeros(3)

    direction = rng.standard_normal(3)
    direction /= np.linalg.norm(direction)
    radius = rng.uniform(MIN_RADIUS, MAX_RADIUS)
    target = start + direction * radius

    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, projection="3d")
    fig.suptitle("RandomSphereOffset — sampled setpoint", fontsize=11)

    sx, sy, sz = sphere_mesh(radius)
    ax.plot_surface(sx, sy, sz, alpha=0.12, color="steelblue", linewidth=0)
    ax.plot_wireframe(sx, sy, sz, alpha=0.08, color="steelblue", linewidth=0.4)

    ax.scatter(*start,  s=60,  color="black",   zorder=5, label="start")
    ax.scatter(*target, s=80,  color="tab:red",  zorder=5, label=f"setpoint  r={radius:.3f} m")
    ax.plot([start[0], target[0]], [start[1], target[1]], [start[2], target[2]],
            lw=1.2, ls="--", color="dimgray")

    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    ax.set_title(f"r = {radius:.3f} m  |  direction = ({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f})",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.set_box_aspect([1, 1, 1])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("random_sphere_offset.png", dpi=150)
    print("Saved random_sphere_offset.png")


if __name__ == "__main__":
    main()
