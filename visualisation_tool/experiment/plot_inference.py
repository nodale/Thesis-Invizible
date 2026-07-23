"""Plot inferred trajectory per champion: XY trajectory and Z over time.

Ground truth = ground_truth[:, 0:3] * POS_NORM  (Vicon, normalised in zarr)
Inferred     = cumulative sum of output[:, 0:3], seeded from ground truth at champion start
Onboard est  = obs[:, 0:3] * POS_NORM
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
SESSION_DIR = Path("/home/nodale/Thesis/Thesis-Invizible/data/real_life/2026-07-22_19-32-47")
POS_NORM    = 3.0   # denormalisation factor for positions and displacements
# ───────────────────────────────────────────────────────────────────────────────


def load(session_dir: Path):
    log = zarr.open(session_dir / "infer_log.zarr", mode="r")
    ground_truth  = np.asarray(log["ground_truth"][:])
    output        = np.asarray(log["output"][:])
    obs           = np.asarray(log["obs"][:])
    generation_id = np.asarray(log["generation_id"][:]).reshape(-1)

    # obs/output may have extra leading rows — align to ground_truth
    n = ground_truth.shape[0]
    obs, output = obs[-n:], output[-n:]

    return {
        "gen_id": generation_id,
        "truth":  ground_truth[:, 0:3] * POS_NORM,
        "output": output[:, 0:3],
        "onboard": obs[:, 0:3] * POS_NORM,
    }


def infer_xyz(output_xyz, snap_pos):
    """Integrate normalised displacements from snap_pos.
    traj[0] = snap_pos, traj[k] = snap_pos + sum(output[0..k-1]) * POS_NORM.
    """
    shifts = np.vstack([[0, 0, 0], np.cumsum(output_xyz[:-1] * POS_NORM, axis=0)])
    return snap_pos + shifts


#color="black", ms=5, mfc="black", mec="black", zorder=3,
#color="red", ms=5, mfc="red", mec="red", zorder=3,
#color="blue", ms=5, mfc="blue", mec="blue", zorder=3,


def plot_champion(axes, output_xyz, truth_xyz, onboard_xyz, champion_id):
    ax_xy, ax_z = axes
    inferred = infer_xyz(output_xyz, truth_xyz[0])
    t = np.arange(len(inferred))

    for ax, xi, yi, zlabel in [(ax_xy, 0, 1, None), (ax_z, None, None, "z")]:
        if zlabel is None:
            ax.plot(inferred[:, xi],  inferred[:, yi],  lw=1.4, label="inferred", color="red", ms=5, mfc="red", mec="red", zorder=3,)
            ax.plot(truth_xyz[:, xi], truth_xyz[:, yi], lw=1.4, ls="solid", label="ground truth (vicon)", color="black", ms=5, mfc="black", mec="black", zorder=3,)
            ax.plot(onboard_xyz[:, xi], onboard_xyz[:, yi], lw=1.4, ls=":", alpha=0.7, label="onboard est", color="blue", ms=5, mfc="blue", mec="blue", zorder=3,)
        else:
            ax.plot(t, inferred[:, 2],   lw=1.4, label="inferred", color="red", ms=5, mfc="red", mec="red", zorder=3,)
            ax.plot(t, truth_xyz[:, 2],  lw=1.4, ls="solid", label="ground truth (vicon)", color="black", ms=5, mfc="black", mec="black", zorder=3,)
            ax.plot(t, onboard_xyz[:, 2], lw=1.4, ls=":", alpha=0.7, label="onboard est", color="blue", ms=5, mfc="blue", mec="blue", zorder=3,)

    ax_xy.set_title(f"Champion {champion_id} — XY")
    ax_xy.set_xlabel("x (m)"); ax_xy.set_ylabel("y (m)")
    ax_xy.legend(fontsize=7); ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.grid(True, alpha=0.1)

    ax_xy.xaxis.set_major_locator(MaxNLocator(8))
    ax_xy.yaxis.set_major_locator(MaxNLocator(8))
    ax_xy.xaxis.set_minor_locator(MaxNLocator(32))
    ax_xy.yaxis.set_minor_locator(MaxNLocator(32))
    ax_xy.grid(which="major", alpha=0.35, linewidth=0.7)
    ax_xy.grid(which="minor", alpha=0.15, linewidth=0.4)

    ax_z.set_title(f"Champion {champion_id} — Z")
    ax_z.set_xlabel("Trajectory Index")
    ax_z.set_ylabel("z (m)")
    ax_z.legend(fontsize=7)

    ax_z.xaxis.set_major_locator(MaxNLocator(8))
    ax_z.yaxis.set_major_locator(MaxNLocator(8))
    ax_z.xaxis.set_minor_locator(MaxNLocator(32))
    ax_z.yaxis.set_minor_locator(MaxNLocator(32))
    ax_z.grid(which="major", alpha=0.35, linewidth=0.7)
    ax_z.grid(which="minor", alpha=0.15, linewidth=0.4)


def main():
    d = load(SESSION_DIR)
    champions = np.unique(d["gen_id"])

    out_dir = SESSION_DIR / "plots"
    out_dir.mkdir(exist_ok=True)

    for cid in champions:
        mask = d["gen_id"] == cid
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        fig.suptitle(f"Champion {cid} — {SESSION_DIR.name}", fontsize=11)
        plot_champion(axes, d["output"][mask], d["truth"][mask], d["onboard"][mask], cid)
        plt.tight_layout()
        fig.savefig(out_dir / f"champion_{cid:02d}.png", dpi=150)
        plt.close(fig)
        print(f"Saved champion_{cid:02d}.png")

    print(f"Done → {out_dir}")


if __name__ == "__main__":
    main()
