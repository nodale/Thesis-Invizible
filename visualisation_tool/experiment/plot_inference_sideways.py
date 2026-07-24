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
SESSION_DIR = Path("/home/nodale/Thesis/Thesis-Invizible/data/real_life/simple_mamba_out6")
ULOG_PATH   = Path("/home/nodale/Thesis/Thesis-Invizible/data/real_life/simple_mamba_out6/")
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


def load_odometry(ulog_path, n_instances=2):
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


def plot_champion_xy(ax, output_xyz, truth_xyz, champion_id, xy_lim, prev_truth_xyz=None, show_labels=True, show_champion_id=True):
    inferred = infer_xyz(output_xyz, truth_xyz[0])

    def _endpoints(xy, color, alpha=1.0):
        ax.plot(*xy[0, :2],  "o", ms=15, mew=5.0, color=color, zorder=5, alpha=alpha)
        ax.plot(*xy[-1, :2], "x", ms=15, mew=5.0, color=color, zorder=5, alpha=alpha)

    if prev_truth_xyz is not None:
        ax.plot(*prev_truth_xyz[:, :2].T, lw=4.0, color="black", alpha=0.18, label="_prev gt")
        _endpoints(prev_truth_xyz, "black", alpha=0.18)

    ax.plot(*truth_xyz[:, :2].T, lw=4.0, color="black", label="ground truth")
    ax.plot(*inferred[:, :2].T,  lw=4.0, color="red",   label="inferred")
    _endpoints(truth_xyz, "black")
    _endpoints(inferred, "red")
    if show_labels:

        from matplotlib.lines import Line2D
        handles, labels = ax.get_legend_handles_labels()
        handles += [
            Line2D([0], [0], marker="o", color="grey", ls="none", ms=15, mew=5.0),
            Line2D([0], [0], marker="x", color="grey", ls="none", ms=15, mew=5.0)
        ]
        labels += ["start", "end"]
        ax.legend(handles=handles, labels=labels, fontsize=7)
    if show_champion_id:
        ax.text(
            0.02, 0.98,
            f"{champion_id}",
            transform=ax.transAxes,
            fontsize=30,
            fontweight="bold",
            va="top",
            ha="left",
            #bbox=dict(
            #    boxstyle="round,pad=0.02",
            #    facecolor="white",
            #    edgecolor="black",
            #    linewidth=1.2,
            #    alpha=0.9
            #),
            zorder=10
        )
    #ax.set_title(f"Champion {champion_id} — GT vs Inferred (XY)")
    #ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    _set_lims(ax, xy_lim, xy_lim); _grid(ax)
    ax.tick_params(
        axis="both",
        which="both",
        labelbottom=False,
        labelleft=False
    )
    #ax.set_axis_off()


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
    #all_xyz = [d["truth"], d["onboard"], *[xyz for _, xyz in (o for o in odo_aligned if o)]]
    all_xyz = [d["truth"], d["onboard"]]
    all_xy  = np.concatenate([a[:, :2] for a in all_xyz])
    all_z   = np.concatenate([a[:, 2]  for a in all_xyz])
    pad     = 0.2
    xy_lim  = (all_xy.min() - pad, all_xy.max() + pad)
    z_lim   = (all_z.min()  - pad, all_z.max()  + pad)
    tc_all  = d["t_s"] - t0
    t_lim   = (tc_all.min(), tc_all.max())

    out_dir = SESSION_DIR / "sideways_plots"
    out_dir.mkdir(exist_ok=True)

    cids = np.unique(d["gen_id"])
    prev_truth = None
    for cid in cids:
        mask = d["gen_id"] == cid
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        #fig.suptitle(f"Champion {cid} — {SESSION_DIR.name}", fontsize=11)
        plot_champion_xy(ax, d["output"][mask], d["truth"][mask], cid, xy_lim, prev_truth_xyz=prev_truth, show_labels=False, show_champion_id=True)
        prev_truth = d["truth"][mask]
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(out_dir / f"champion_{cid}.png", dpi=150, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
        print(f"Saved champion_{cid}.png")

    #plotting the legend
    from matplotlib.lines import Line2D

    legend_fig = plt.figure(figsize=(3, 1))
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.axis("off")

    legend_elements = [
        Line2D([0], [0],
               color="black",
               lw=8.0,
               label="ground truth"),

        Line2D([0], [0],
               color="red",
               lw=8.0,
               label="inferred"),

        Line2D([0], [0],
               color="black",
               lw=8.0,
               alpha=0.18,
               label="previous ground truth"),

        Line2D([0], [0],
               marker="o",
               color="grey",
               ls="none",
               ms=15, mew=5.0,
               label="start"),

        Line2D([0], [0],
               marker="x",
               color="grey",
               ls="none",
               ms=15, mew=5.0,
               label="end"),
    ]

    legend_ax.legend(handles=legend_elements,
                     loc="center",
                     ncol=5,
                     fontsize=9,
                     frameon=False)

    legend_fig.savefig(out_dir / "legend.png",
                       dpi=150,
                       bbox_inches="tight",
                       pad_inches=0.03)
    plt.close(legend_fig)

    print(f"Done → {out_dir}")


if __name__ == "__main__":
    main()
