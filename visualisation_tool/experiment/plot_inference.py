"""Plot inferred trajectory per champion: three panels per champion.

Left:   inferred vs ground truth (XY)
Middle: est_odometry[0..2] vs ground truth (XY)
Right:  Z over time for all signals
Axes are fixed across all champions for easy comparison.
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


def load_zarr(session_dir):
    log = zarr.open(session_dir / "infer_log.zarr", mode="r")
    gt  = np.asarray(log["ground_truth"][:])
    out = np.asarray(log["output"][:])
    obs = np.asarray(log["obs"][:])
    gen = np.asarray(log["generation_id"][:]).reshape(-1)
    t_s = np.asarray(log["state_arrival_ns"][:]).reshape(-1).astype(np.float64) / 1e9
    n   = gt.shape[0]
    return {"gen_id": gen, "truth": gt[:, :3] * POS_NORM,
            "output": out[-n:, :3], "onboard": obs[-n:, :3] * POS_NORM, "t_s": t_s}


def load_odometry(ulog_path, n_instances=3):
    by_id = {d.multi_id: d for d in ULog(str(ulog_path)).data_list
             if d.name == "estimator_odometry"}
    result = []
    for i in range(n_instances):
        if i not in by_id:
            result.append(None)
        else:
            d = by_id[i]
            result.append((d.data["timestamp"].astype(np.float64) / 1e6,
                           np.column_stack([d.data[f"position[{j}]"] for j in range(3)])))
    return result


def align_clocks(t_zarr, z_zarr, odo_list):
    """Affine-fit ulog clock → zarr clock using est_odometry[0] Z ≡ ground-truth Z."""
    t_odo, xyz = odo_list[0]
    f   = interp1d(t_zarr, z_zarr, bounds_error=False, fill_value=np.nan)
    dt  = 0.1
    lag = (np.argmax(correlate(
        interp1d(t_zarr, z_zarr)(np.arange(t_zarr[0], t_zarr[-1], dt)) - z_zarr.mean(),
        interp1d(t_odo,  xyz[:, 2])(np.arange(t_odo[0],  t_odo[-1],  dt)) - xyz[:, 2].mean(),
        mode="full")) - (len(np.arange(t_odo[0], t_odo[-1], dt)) - 1)) * dt

    def mse(p):
        t = t_odo * p[1] + p[0]
        z = f(t);  m = np.isfinite(z)
        return np.mean((z[m] - xyz[m, 2])**2) if m.sum() > 5 else 1e9

    res = minimize(mse, [lag, 1.0], method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 20000})
    return res.x  # (offset, scale)


def infer_xyz(output_xyz, snap_pos):
    shifts = np.vstack([[0, 0, 0], np.cumsum(output_xyz[:-1] * POS_NORM, axis=0)])
    return snap_pos + shifts


def _grid(ax):
    for which, a, lw in [("major", 0.35, 0.7), ("minor", 0.15, 0.4)]:
        ax.grid(which=which, alpha=a, linewidth=lw)
    ax.xaxis.set_major_locator(MaxNLocator(8));  ax.xaxis.set_minor_locator(MaxNLocator(32))
    ax.yaxis.set_major_locator(MaxNLocator(8));  ax.yaxis.set_minor_locator(MaxNLocator(32))


def _set_lims(ax, xlim, ylim):
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)


ODO_COLORS = ["tab:green", "tab:orange", "tab:purple"]


def plot_champion(axes, output_xyz, truth_xyz, onboard_xyz, tc, odo_list,
                  champion_id, xy_lim, z_lim, t_lim):
    ax_inf, ax_odo, ax_z = axes
    inferred = infer_xyz(output_xyz, truth_xyz[0])
    odom_name= ["EKF2+EV", "EKF2+EV every 32 seconds", "EKF2"]

    def _clip(odo):
        t, xyz = odo
        m = (t >= tc[0]) & (t <= tc[-1])
        return t[m], xyz[m]

    def _endpoints(ax, xy, color):
        ax.plot(*xy[0, :2],  "o", ms=5, color=color, zorder=5)
        ax.plot(*xy[-1, :2], "s", ms=5, color=color, zorder=5)

    def _add_endpoint_legend(ax):
        from matplotlib.lines import Line2D
        handles, labels = ax.get_legend_handles_labels()
        handles += [Line2D([0], [0], marker="o", color="grey", ls="none", ms=5),
                    Line2D([0], [0], marker="s", color="grey", ls="none", ms=5)]
        labels  += ["start", "end"]
        ax.legend(handles=handles, labels=labels, fontsize=7)

    # left — inferred vs GT
    ax_inf.plot(*truth_xyz[:, :2].T, lw=1.4, color="black", label="ground truth")
    ax_inf.plot(*inferred[:, :2].T,  lw=1.4, color="red",   label="inferred")
    _endpoints(ax_inf, truth_xyz, "black"); _endpoints(ax_inf, inferred, "red")
    ax_inf.set_title(f"Champion {champion_id} — Inferred vs GT")
    ax_inf.set_xlabel("x (m)"); ax_inf.set_ylabel("y (m)")
    _add_endpoint_legend(ax_inf); ax_inf.set_aspect("equal", adjustable="box")
    _set_lims(ax_inf, xy_lim, xy_lim); _grid(ax_inf)

    # middle — odometry vs GT
    ax_odo.plot(*truth_xyz[:, :2].T, lw=1.4, color="black", label="ground truth")
    _endpoints(ax_odo, truth_xyz, "black")
    for i, odo in enumerate(odo_list):
        if odo is None: continue
        t_w, xyz_w = _clip(odo)
        if len(t_w):
            ax_odo.plot(*xyz_w[:, :2].T, lw=1.4, ls="--", color=ODO_COLORS[i], label=odom_name[i])
            _endpoints(ax_odo, xyz_w, ODO_COLORS[i])
    ax_odo.set_title(f"Champion {champion_id} — Odometry vs GT")
    ax_odo.set_xlabel("x (m)"); ax_odo.set_ylabel("y (m)")
    _add_endpoint_legend(ax_odo); ax_odo.set_aspect("equal", adjustable="box")
    _set_lims(ax_odo, xy_lim, xy_lim); _grid(ax_odo)

    # right — Z over time
    ax_z.plot(tc, truth_xyz[:, 2],   lw=1.4, color="black",     label="ground truth")
    ax_z.plot(tc, inferred[:, 2],    lw=1.4, color="red",        label="inferred")
    ax_z.plot(tc, onboard_xyz[:, 2], lw=1.4, color="blue", ls=":", alpha=0.7, label="onboard est")
    for i, odo in enumerate(odo_list):
        if odo is None: continue
        t_w, xyz_w = _clip(odo)
        if len(t_w):
            ax_z.plot(t_w, xyz_w[:, 2], lw=1.4, ls="--", alpha=0.7, color=ODO_COLORS[i], label=odom_name[i])
    ax_z.set_title(f"Champion {champion_id} — Z"); ax_z.set_xlabel("time (s)"); ax_z.set_ylabel("z (m)")
    ax_z.legend(fontsize=7)
    _set_lims(ax_z, None, z_lim); _grid(ax_z)


def main():
    d        = load_zarr(SESSION_DIR)
    odo_list = load_odometry(ULOG_PATH)

    print("Aligning clocks...")
    offset, scale = align_clocks(d["t_s"], d["truth"][:, 2], odo_list)
    print(f"  offset={offset:.4f} s, scale={scale:.7f}")

    t0 = d["t_s"].min()
    odo_aligned = []
    for odo in odo_list:
        if odo is None:
            odo_aligned.append(None)
        else:
            t, xyz = odo
            odo_aligned.append((t * scale + offset - t0, xyz))

    # compute global axis limits from all data
    all_xyz = [d["truth"], d["onboard"],
               *[xyz for _, xyz in (o for o in odo_aligned if o)]]
    all_xy  = np.concatenate([a[:, :2] for a in all_xyz])
    all_z   = np.concatenate([a[:, 2]  for a in all_xyz])
    pad     = 0.5
    xy_lim  = (all_xy.min() - pad, all_xy.max() + pad)
    z_lim   = (all_z.min()  - pad, all_z.max()  + pad)
    tc_all  = d["t_s"] - t0
    t_lim   = (tc_all.min(), tc_all.max())

    out_dir = SESSION_DIR / "plots"
    out_dir.mkdir(exist_ok=True)

    for cid in np.unique(d["gen_id"]):
        mask = d["gen_id"] == cid
        tc   = d["t_s"][mask] - t0
        fig, axes = plt.subplots(1, 3, figsize=(20, 5), gridspec_kw={"wspace": 0.25},)
        fig.suptitle(f"Champion {cid} — {SESSION_DIR.name}", fontsize=11)
        plot_champion(axes, d["output"][mask], d["truth"][mask], d["onboard"][mask],
                      tc, odo_aligned, cid, xy_lim, z_lim, t_lim)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out_dir / f"champion_{cid:02d}.png", dpi=150)
        plt.close(fig)
        print(f"Saved champion_{cid:02d}.png")

    print(f"Done → {out_dir}")


if __name__ == "__main__":
    main()
