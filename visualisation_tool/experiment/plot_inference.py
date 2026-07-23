"""Plot inferred trajectory per champion: XY trajectory and Z over time.

Ground truth = ground_truth[:, 0:3] * POS_NORM  (Vicon, normalised in zarr)
Inferred     = cumulative sum of output[:, 0:3], seeded from ground truth at champion start
Onboard est  = obs[:, 0:3] * POS_NORM
estimator_odometry[0/1/2] = PX4 ulog, clock-aligned to zarr via affine fit on Z
"""

import zarr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
from pyulog import ULog
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.signal import correlate

# ── Config ─────────────────────────────────────────────────────────────────────
SESSION_DIR = Path("/home/nodale/Thesis/Thesis-Invizible/data/real_life/2026-07-22_19-32-47")
ULOG_PATH   = Path("/home/nodale/Thesis/Thesis-Invizible/data/thesis_px4_logs/log_201_2026-7-22-19-37-46.ulg")
POS_NORM    = 3.0
# ───────────────────────────────────────────────────────────────────────────────


def load_zarr(session_dir: Path):
    log = zarr.open(session_dir / "infer_log.zarr", mode="r")
    ground_truth  = np.asarray(log["ground_truth"][:])
    output        = np.asarray(log["output"][:])
    obs           = np.asarray(log["obs"][:])
    generation_id = np.asarray(log["generation_id"][:]).reshape(-1)
    t_s           = np.asarray(log["state_arrival_ns"][:]).reshape(-1).astype(np.float64) / 1e9
    n = ground_truth.shape[0]
    obs, output = obs[-n:], output[-n:]
    return {
        "gen_id":  generation_id,
        "truth":   ground_truth[:, 0:3] * POS_NORM,
        "output":  output[:, 0:3],
        "onboard": obs[:, 0:3] * POS_NORM,
        "t_s":     t_s,
    }


def load_odometry(ulog_path: Path, n_instances: int = 3):
    """Return list of (t_s, xyz) for estimator_odometry instances 0..n-1."""
    ulog  = ULog(str(ulog_path))
    by_id = {d.multi_id: d for d in ulog.data_list if d.name == "estimator_odometry"}
    result = []
    for i in range(n_instances):
        if i not in by_id:
            result.append(None)
        else:
            d   = by_id[i]
            t_s = d.data["timestamp"].astype(np.float64) / 1e6  # µs → s
            xyz = np.column_stack([d.data[f"position[{j}]"] for j in range(3)])
            result.append((t_s, xyz))
    return result


def align_clocks(t_zarr_s, z_zarr, odo_list):
    """Affine-fit the ulog clock to the zarr clock using est_odometry[0] Z,
    which measures the same quantity as ground-truth Z (Vicon).
    Returns (offset, scale) such that t_odo_aligned = t_odo * scale + offset.
    """
    odo0 = odo_list[0]
    assert odo0 is not None, "est_odometry[0] required for clock alignment"
    t_odo, xyz_odo = odo0
    f = interp1d(t_zarr_s, z_zarr, bounds_error=False, fill_value=np.nan)

    # coarse offset via cross-correlation
    dt = 0.1
    ta = np.arange(t_zarr_s[0], t_zarr_s[-1], dt)
    tb = np.arange(t_odo[0], t_odo[-1], dt)
    lag0 = (np.argmax(correlate(
        interp1d(t_zarr_s, z_zarr)(ta) - z_zarr.mean(),
        interp1d(t_odo, xyz_odo[:, 2])(tb) - xyz_odo[:, 2].mean(),
        mode="full",
    )) - (len(tb) - 1)) * dt

    def mse(params):
        off, scale = params
        t_sh = t_odo * scale + off
        z_gt = f(t_sh)
        mask = np.isfinite(z_gt)
        return np.mean((z_gt[mask] - xyz_odo[mask, 2]) ** 2) if mask.sum() > 5 else 1e9

    res = minimize(mse, x0=[lag0, 1.0], method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 20000})
    return res.x  # (offset, scale)


def infer_xyz(output_xyz, snap_pos):
    shifts = np.vstack([[0, 0, 0], np.cumsum(output_xyz[:-1] * POS_NORM, axis=0)])
    return snap_pos + shifts


def _grid(ax):
    ax.xaxis.set_major_locator(MaxNLocator(8));  ax.xaxis.set_minor_locator(MaxNLocator(32))
    ax.yaxis.set_major_locator(MaxNLocator(8));  ax.yaxis.set_minor_locator(MaxNLocator(32))
    ax.grid(which="major", alpha=0.35, linewidth=0.7)
    ax.grid(which="minor", alpha=0.15, linewidth=0.4)


ODO_COLORS = ["tab:green", "tab:orange", "tab:purple"]


def plot_champion(axes, output_xyz, truth_xyz, onboard_xyz, t_s, odo_list, champion_id, t0_s):
    ax_xy, ax_z = axes
    inferred = infer_xyz(output_xyz, truth_xyz[0])
    tc = t_s - t0_s  # seconds from session start

    def _plot_zarr(xyz, **kw):
        ax_xy.plot(xyz[:, 0], xyz[:, 1], lw=1.4, **kw)
        ax_z.plot(tc, xyz[:, 2], lw=1.4, **kw)

    _plot_zarr(inferred,    color="red",   label="inferred")
    _plot_zarr(truth_xyz,   color="black", label="ground truth (vicon)")
    _plot_zarr(onboard_xyz, color="blue",  ls=":", alpha=0.7, label="onboard est")

    for i, odo in enumerate(odo_list):
        if odo is None:
            continue
        t_odo_s, xyz_odo = odo
        t_odo_c = t_odo_s - t0_s
        mask = (t_odo_c >= tc[0]) & (t_odo_c <= tc[-1])
        if mask.sum() == 0:
            continue
        ax_xy.plot(xyz_odo[mask, 0], xyz_odo[mask, 1], lw=1.4,
                   color=ODO_COLORS[i], ls="--", alpha=0.7, label=f"est_odometry[{i}]")
        ax_z.plot(t_odo_c[mask], xyz_odo[mask, 2], lw=1.4,
                  color=ODO_COLORS[i], ls="--", alpha=0.7, label=f"est_odometry[{i}]")

    ax_xy.set_title(f"Champion {champion_id} — XY")
    ax_xy.set_xlabel("x (m)"); ax_xy.set_ylabel("y (m)")
    ax_xy.legend(fontsize=7); 
    #ax_xy.set_aspect("equal", adjustable="datalim")
    _grid(ax_xy)
    ax_xy.set_xlim(-2.0, 2.0)
    ax_xy.set_ylim(-2.0, 2.0)


    ax_z.set_title(f"Champion {champion_id} — Z")
    ax_z.set_xlabel("time (s)"); ax_z.set_ylabel("z (m)")
    ax_z.legend(fontsize=7)
    _grid(ax_z)


def main():
    d        = load_zarr(SESSION_DIR)
    odo_list = load_odometry(ULOG_PATH)

    print("Aligning clocks via Z cross-correlation + affine fit...")
    offset, scale = align_clocks(d["t_s"], d["truth"][:, 2], odo_list)
    print(f"  offset={offset:.4f} s, scale={scale:.7f}")

    odo_aligned = []
    for odo in odo_list:
        if odo is None:
            odo_aligned.append(None)
        else:
            t_odo, xyz_odo = odo
            odo_aligned.append((t_odo * scale + offset, xyz_odo))

    t0_s    = d["t_s"].min()
    out_dir = SESSION_DIR / "plots"
    out_dir.mkdir(exist_ok=True)

    for cid in np.unique(d["gen_id"]):
        mask = d["gen_id"] == cid
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        fig.suptitle(f"Champion {cid} — {SESSION_DIR.name}", fontsize=11)
        plot_champion(axes, d["output"][mask], d["truth"][mask], d["onboard"][mask],
                      d["t_s"][mask], odo_aligned, cid, t0_s)
        plt.tight_layout()
        fig.savefig(out_dir / f"champion_{cid:02d}.png", dpi=150)
        plt.close(fig)
        print(f"Saved champion_{cid:02d}.png")

    print(f"Done → {out_dir}")


if __name__ == "__main__":
    main()
