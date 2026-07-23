"""Visualize online_learner.py recorded data: infer_log.zarr, online_data.zarr, logs/.

Plots written to SESSION_DIR/viz/:
  00_current_champion_estimation.png  – current champion's most recent lap
  01_inference_trace.png              – recent pos/vel from obs
  02_episode_trajectories_xy.png      – vicon-supervised episode XY paths
  03_swap_events.png                  – accepted swap positions
  04_cycle_ate.png                    – per-cycle champion vs challenger ATE
  05_run<r>_lap<l>_gen<g>_round<r>_estimation.png  – per-lap 3-panel figure

Usage:
    conda activate alpha311 && python visualize_v2.py
"""

import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import zarr

# ── config ────────────────────────────────────────────────────────────────────
SESSION_DIR = Path("/home/nodale/Thesis/Thesis-Invizible/data/real_life/cls_out10")
ULOG_PATH   = Path("/home/nodale/Thesis/Thesis-Invizible/data/real_life/cls_out10/log_202_2026-7-22-21-22-22.ulg")
SUMMARY_ONLY = False

POS_NORM   = 3.0
VEL_NORM   = 0.8
NORM = dict(pos=POS_NORM, vel=VEL_NORM, angvel=0.5, acc=25.0, thrust=9.81, setpoint=3.0)
PARTIAL_TOLERANCE_S  = 2.0
MIN_CURRENT_PARTIAL_S = 1.0
ODO_COLORS = ["tab:green", "tab:orange", "tab:purple"]
# ─────────────────────────────────────────────────────────────────────────────


# ── data loading ──────────────────────────────────────────────────────────────

def load_zarr(session_dir: Path) -> dict:
    log = zarr.open(str(session_dir / "infer_log.zarr"), mode="r")
    obs = np.asarray(log["obs"][:])
    out = np.asarray(log["output"][:])
    swap_events = np.asarray(log["swap_events"][:]) if "swap_events" in log else np.zeros((0, 7))

    if "ground_truth" in log:
        gt  = np.asarray(log["ground_truth"][:])
        gen = np.asarray(log["generation_id"][:]).reshape(-1)
        # state_arrival_ns preferred; fall back to timestamp_ns for older logs
        key = "state_arrival_ns" if "state_arrival_ns" in log else "timestamp_ns"
        t_s = np.asarray(log[key][:]).reshape(-1).astype(np.float64) / 1e9
        n   = gt.shape[0]
        obs, out = obs[-n:], out[-n:]
    else:
        gt  = np.zeros((0, 6))
        gen = np.zeros(0, dtype=np.int32)
        t_s = np.zeros(0, dtype=np.float64)

    return dict(
        obs=obs, output=out[:, :3], swap_events=swap_events,
        truth=gt[:, :3] * POS_NORM,
        gt_full=gt,
        onboard=obs[:, :3] * POS_NORM,
        gen_id=gen,
        t_s=t_s,
    )


def load_episodes(session_dir: Path) -> np.ndarray:
    root = zarr.open(str(session_dir / "online_data.zarr"), mode="r")
    return np.asarray(root["episodes"])


def load_odometry(ulog_path: Path, n_instances: int = 2):
    """Return list of (t_s, xyz) per estimator_odometry instance, or None."""
    try:
        from pyulog import ULog
    except ImportError:
        return [None] * n_instances
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
    from scipy.interpolate import interp1d
    from scipy.optimize import minimize
    from scipy.signal import correlate

    t_odo, xyz = odo_list[0]
    dt  = 0.1
    lag = (np.argmax(correlate(
        interp1d(t_zarr, z_zarr)(np.arange(t_zarr[0], t_zarr[-1], dt)) - z_zarr.mean(),
        interp1d(t_odo,  xyz[:, 2])(np.arange(t_odo[0], t_odo[-1], dt)) - xyz[:, 2].mean(),
        mode="full")) - (len(np.arange(t_odo[0], t_odo[-1], dt)) - 1)) * dt

    f = interp1d(t_zarr, z_zarr, bounds_error=False, fill_value=np.nan)

    def mse(p):
        t = t_odo * p[1] + p[0]
        z = f(t);  m = np.isfinite(z)
        return np.mean((z[m] - xyz[m, 2]) ** 2) if m.sum() > 5 else 1e9

    res = minimize(mse, [lag, 1.0], method="Nelder-Mead",
                   options={"xatol": 1e-4, "fatol": 1e-8, "maxiter": 20000})
    return res.x


def parse_cycle_ate(log_path: Path):
    if not log_path.exists():
        return []
    rows = []
    for m in re.finditer(r"cur_ATE=([\d.]+)\s+new_ATE=([\d.]+)", log_path.read_text()):
        cur, new = float(m.group(1)), float(m.group(2))
        rows.append((cur, new, new < cur))
    return rows


def figure_eight_period_s(default=48.3):
    try:
        from alphacore.alpha.providers.trajectory import EIGHT_OMEGA
        return 2 * math.pi / EIGHT_OMEGA
    except Exception:
        return default


# ── segmentation ──────────────────────────────────────────────────────────────

def run_starts(gen_id):
    if len(gen_id) == 0:
        return [0]
    resets = np.flatnonzero((gen_id[1:] == 0) & (gen_id[:-1] != 0)) + 1
    return [0, *resets.tolist()]


def dwell_segments(gen_id):
    """(gen, lo, hi) for each contiguous block of the same generation_id."""
    if len(gen_id) == 0:
        return []
    edges  = np.flatnonzero(np.diff(gen_id)) + 1
    bounds = [0, *edges.tolist(), len(gen_id)]
    return [(int(gen_id[a]), a, b) for a, b in zip(bounds, bounds[1:])]


def lap_segments(gen_id, t_s, period_s):
    """One entry per figure-8 lap actually flown, with partial-lap handling."""
    laps     = []
    segments = dwell_segments(gen_id)
    for seg_i, (gen, seg_lo, seg_hi) in enumerate(segments):
        is_last = seg_i == len(segments) - 1
        lo, round_idx = seg_lo, 0
        while lo < seg_hi - 1:
            elapsed = t_s[lo:seg_hi] - t_s[lo]
            hi      = lo + max(int(np.searchsorted(elapsed, period_s, side="right")), 1)
            if hi - lo < 2:
                break
            span_s    = elapsed[hi - lo - 1]
            partial   = span_s < period_s - PARTIAL_TOLERANCE_S
            is_final  = hi >= seg_hi - 1
            if partial and not (is_last and is_final and span_s >= MIN_CURRENT_PARTIAL_S):
                break
            laps.append(dict(gen=gen, round_idx=round_idx, lo=lo, hi=hi, partial=partial))
            lo, round_idx = hi, round_idx + 1
    return laps


# ── plot helpers ──────────────────────────────────────────────────────────────

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


def _endpoints(ax, xy, color):
    ax.plot(*xy[0,  :2], "o", ms=5, color=color, zorder=5)
    ax.plot(*xy[-1, :2], "s", ms=5, color=color, zorder=5)


def _add_endpoint_legend(ax):
    from matplotlib.lines import Line2D
    handles, labels = ax.get_legend_handles_labels()
    handles += [Line2D([0], [0], marker="o", color="grey", ls="none", ms=5),
                Line2D([0], [0], marker="s", color="grey", ls="none", ms=5)]
    labels  += ["start", "end"]
    ax.legend(handles=handles, labels=labels, fontsize=7)


def infer_xyz(output_xyz, snap_pos):
    """Integrate model output deltas from snap_pos."""
    shifts = np.vstack([[0, 0, 0], np.cumsum(output_xyz[:-1] * POS_NORM, axis=0)])
    return snap_pos + shifts


# ── individual plots ──────────────────────────────────────────────────────────

def plot_inference_trace(d: dict, viz_dir: Path, n_last=6000):
    obs = d["obs"][-n_last:]
    n   = obs.shape[0]
    pos = obs[:, :3] * NORM["pos"]
    vel = obs[:, 3:6] * NORM["vel"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for i, label in enumerate("xyz"):
        axes[0].plot(pos[:, i], label=f"pos {label}", linewidth=0.8)
        axes[1].plot(vel[:, i], label=f"vel {label}", linewidth=0.8)
    axes[0].set_ylabel("position (m)")
    axes[1].set_ylabel("velocity (m/s)")
    axes[1].set_xlabel(f"sample (last {n} shown, champion inference log)")
    for ax in axes:
        ax.legend(loc="upper right", fontsize=8)
        _grid(ax)
    fig.suptitle("Champion inference log: obs position / velocity (recent window)")
    fig.tight_layout()
    fig.savefig(viz_dir / "01_inference_trace.png", dpi=130)
    plt.close(fig)


def plot_trajectory_xy(episodes: np.ndarray, viz_dir: Path, n_episodes=20):
    fig, ax = plt.subplots(figsize=(7, 7))
    idxs = np.linspace(0, episodes.shape[0] - 1,
                       min(n_episodes, episodes.shape[0])).astype(int)
    cmap = plt.get_cmap("viridis")
    for i, idx in enumerate(idxs):
        pos = episodes[idx, :, :3] * NORM["pos"]
        ax.plot(pos[:, 0], pos[:, 1],
                color=cmap(i / max(len(idxs) - 1, 1)), alpha=0.7, linewidth=1)
        _endpoints(ax, pos, cmap(i / max(len(idxs) - 1, 1)))
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Vicon-supervised episode trajectories (XY), "
                 f"{len(idxs)} of {episodes.shape[0]} episodes shown\n"
                 "(color = episode order, early→late)")
    ax.set_aspect("equal", adjustable="datalim")
    _grid(ax)
    fig.tight_layout()
    fig.savefig(viz_dir / "02_episode_trajectories_xy.png", dpi=130)
    plt.close(fig)


def plot_swap_events(d: dict, viz_dir: Path):
    swap_events = d["swap_events"]
    if len(swap_events) == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    x, y, gen_idx = swap_events[:, 4], swap_events[:, 5], swap_events[:, 2]
    sc = ax.scatter(x, y, c=gen_idx, cmap="plasma", s=120, edgecolor="k", zorder=3)
    for xi, yi, g in zip(x, y, gen_idx):
        ax.annotate(f"gen {int(g)}", (xi, yi),
                    textcoords="offset points", xytext=(8, 8), fontsize=9)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Accepted champion→challenger swap events ({len(swap_events)} total)\n"
                 "position at moment of swap, colored by generation index")
    plt.colorbar(sc, ax=ax, label="generation index")
    _grid(ax)
    fig.tight_layout()
    fig.savefig(viz_dir / "03_swap_events.png", dpi=130)
    plt.close(fig)


def plot_cycle_ate(rows: list, viz_dir: Path):
    if not rows:
        print("[viz] no cur_ATE/new_ATE lines found in the log -- skipping cycle ATE plot")
        return
    cycles   = np.arange(len(rows))
    cur      = [r[0] for r in rows]
    new      = [r[1] for r in rows]
    accepted = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    ax.bar(cycles - width / 2, cur, width, label="champion (current)", color="#888888")
    ax.bar(cycles + width / 2, new, width, label="challenger (new)",
           color=["#2ca02c" if a else "#d62728" for a in accepted])
    for i, a in enumerate(accepted):
        ax.annotate("SWAPPED" if a else "rejected", (i + width / 2, new[i]),
                    textcoords="offset points", xytext=(0, 6),
                    ha="center", fontsize=8, color="#2ca02c" if a else "#d62728")
    ax.set_xticks(cycles)
    ax.set_xticklabels([f"cycle {i}" for i in cycles])
    ax.set_ylabel("ATE (m, lower is better)")
    ax.set_title("Per-cycle champion vs. challenger evaluation (ATE)")
    ax.legend()
    _grid(ax)
    fig.tight_layout()
    fig.savefig(viz_dir / "04_cycle_ate.png", dpi=130)
    plt.close(fig)


def _plot_lap(axes, d: dict, lap: dict, odo_list, period_s: float,
              xy_lim, z_lim, title_prefix: str):
    """Three-panel lap figure: inferred vs GT (XY) | odometry vs GT (XY) | Z over time."""
    lo, hi   = lap["lo"], lap["hi"]
    partial  = lap["partial"]
    t_s      = d["t_s"]
    t0       = t_s[lo]
    tc       = t_s[lo:hi] - t0
    duration = tc[-1] if len(tc) else 0.0

    truth_xyz   = d["truth"][lo:hi]
    onboard_xyz = d["onboard"][lo:hi]
    output_xyz  = d["output"][lo:hi]
    inferred    = infer_xyz(output_xyz, truth_xyz[0])

    tag   = " — PARTIAL" if partial else ""
    title = (f"{title_prefix}gen {lap['gen']} round {lap['round_idx']} — "
             f"{duration:.1f}s of {period_s:.1f}s ({hi - lo} ticks){tag}")

    odom_name = ["EKF2+EV", "EKF2+EV every 32 s", "EKF2"]

    def _clip(odo):
        t, xyz = odo
        m = (t >= t_s[lo] - t0) & (t <= t_s[hi - 1] - t0)
        return t[m], xyz[m]

    ax_inf, ax_odo, ax_z = axes

    # left – inferred vs ground truth (XY)
    ax_inf.plot(*truth_xyz[:, :2].T,  lw=1.4, color="black", label="ground truth")
    ax_inf.plot(*inferred[:, :2].T,   lw=1.4, color="red",   label="inferred")
    ax_inf.plot(*onboard_xyz[:, :2].T, lw=1.0, color="blue",
                ls=":", alpha=0.6, label="onboard est")
    _endpoints(ax_inf, truth_xyz,  "black")
    _endpoints(ax_inf, inferred,   "red")
    ax_inf.set_title(f"{title}\nInferred vs GT")
    ax_inf.set_xlabel("x (m)"); ax_inf.set_ylabel("y (m)")
    ax_inf.set_aspect("equal", adjustable="box")
    _add_endpoint_legend(ax_inf); _set_lims(ax_inf, xy_lim, xy_lim); _grid(ax_inf)

    # middle – odometry vs ground truth (XY)
    ax_odo.plot(*truth_xyz[:, :2].T, lw=1.4, color="black", label="ground truth")
    _endpoints(ax_odo, truth_xyz, "black")
    for i, odo in enumerate(odo_list):
        if odo is None:
            continue
        t_w, xyz_w = _clip(odo)
        if len(t_w):
            ax_odo.plot(*xyz_w[:, :2].T, lw=1.4, ls="--",
                        color=ODO_COLORS[i], label=odom_name[i])
            _endpoints(ax_odo, xyz_w, ODO_COLORS[i])
    ax_odo.set_title("Odometry vs GT")
    ax_odo.set_xlabel("x (m)"); ax_odo.set_ylabel("y (m)")
    ax_odo.set_aspect("equal", adjustable="box")
    _add_endpoint_legend(ax_odo); _set_lims(ax_odo, xy_lim, xy_lim); _grid(ax_odo)

    # right – Z over time
    ax_z.plot(tc, truth_xyz[:, 2],   lw=1.4, color="black",     label="ground truth")
    ax_z.plot(tc, inferred[:, 2],    lw=1.4, color="red",        label="inferred")
    ax_z.plot(tc, onboard_xyz[:, 2], lw=1.0, color="blue", ls=":", alpha=0.7,
              label="onboard est")
    for i, odo in enumerate(odo_list):
        if odo is None:
            continue
        t_w, xyz_w = _clip(odo)
        if len(t_w):
            ax_z.plot(t_w, xyz_w[:, 2], lw=1.2, ls="--", alpha=0.7,
                      color=ODO_COLORS[i], label=odom_name[i])
    ax_z.set_title("Z over time")
    ax_z.set_xlabel("time (s)"); ax_z.set_ylabel("z (m)")
    ax_z.legend(fontsize=7)
    _set_lims(ax_z, None, z_lim); _grid(ax_z)

    err = np.linalg.norm(inferred - truth_xyz, axis=1)
    return err, duration


def plot_generation_estimation(d: dict, odo_list: list, viz_dir: Path,
                                run_id: int, xy_lim, z_lim):
    """One 3-panel figure per lap in this run."""
    period_s = figure_eight_period_s()
    laps     = lap_segments(d["gen_id"], d["t_s"], period_s)
    if not laps:
        print("[viz] no generation_id data -- skipping per-lap estimation plots")
        return
    for lap_i, lap in enumerate(laps):
        fig, axes = plt.subplots(1, 3, figsize=(20, 5), gridspec_kw={"wspace": 0.25})
        fig.suptitle(f"run {run_id} — lap {lap_i} "
                     f"(gen {lap['gen']} round {lap['round_idx']})", fontsize=11)
        err, dur = _plot_lap(axes, d, lap, odo_list, period_s, xy_lim, z_lim,
                              title_prefix=f"run {run_id}: ")
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        fname = (f"05_run{run_id}_lap{lap_i:03d}"
                 f"_gen{lap['gen']:03d}_round{lap['round_idx']:02d}_estimation.png")
        fig.savefig(viz_dir / fname, dpi=130)
        plt.close(fig)
        status = "PARTIAL" if lap["partial"] else "full lap"
        print(f"[viz] run {run_id} lap {lap_i} (gen {lap['gen']} round {lap['round_idx']}): "
              f"{dur:.1f}s ({status}), mean estimate error {err.mean():.3f}m")


def plot_current_champion_estimation(d: dict, odo_list: list, viz_dir: Path,
                                      run_id: int, xy_lim, z_lim):
    """Single 3-panel figure for the most recent lap (current champion)."""
    period_s = figure_eight_period_s()
    laps     = lap_segments(d["gen_id"], d["t_s"], period_s)
    if not laps:
        print("[viz] no generation_id data -- skipping current-champion plot")
        return
    lap = laps[-1]
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), gridspec_kw={"wspace": 0.25})
    fig.suptitle(f"CURRENT champion — run {run_id} "
                 f"gen {lap['gen']} round {lap['round_idx']}", fontsize=11)
    err, dur = _plot_lap(axes, d, lap, odo_list, period_s, xy_lim, z_lim,
                          title_prefix=f"CURRENT champion (run {run_id}) — ")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(viz_dir / "00_current_champion_estimation.png", dpi=130)
    plt.close(fig)
    print(f"[viz] current champion (run {run_id}, gen {lap['gen']} round {lap['round_idx']}): "
          f"{dur:.1f}s, mean estimate error {err.mean():.3f}m")


def summarize_final_generation_error(d: dict, run_id: int):
    period_s = figure_eight_period_s()
    laps     = [lap for lap in lap_segments(d["gen_id"], d["t_s"], period_s)
                if not lap["partial"]]
    if not laps:
        return None
    final_gen  = laps[-1]["gen"]
    final_laps = [lap for lap in laps if lap["gen"] == final_gen]
    errs = np.concatenate([
        np.linalg.norm(
            infer_xyz(d["output"][lap["lo"]:lap["hi"]], d["truth"][lap["lo"]]) - d["truth"][lap["lo"]:lap["hi"]],
            axis=1)
        for lap in final_laps
    ])
    return dict(run_id=run_id, generation=int(final_gen), n_laps=len(final_laps),
                mean_err=float(errs.mean()), max_err=float(errs.max()))


# ── global axis limits ────────────────────────────────────────────────────────

def compute_global_limits(d: dict, odo_list: list):
    all_xyz = [d["truth"], d["onboard"]]
    for odo in odo_list:
        if odo is not None:
            all_xyz.append(odo[1])
    all_xy = np.concatenate([a[:, :2] for a in all_xyz])
    all_z  = np.concatenate([a[:,  2] for a in all_xyz])
    pad    = 0.2
    return (all_xy.min() - pad, all_xy.max() + pad), (all_z.min() - pad, all_z.max() + pad)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    viz_dir = SESSION_DIR / "viz"
    viz_dir.mkdir(exist_ok=True)

    print(f"[viz] loading recorded data from {SESSION_DIR}...")
    d        = load_zarr(SESSION_DIR)
    episodes = load_episodes(SESSION_DIR)

    log_dir       = SESSION_DIR / "logs"
    log_candidates = sorted(log_dir.glob("online_learner*.log"),
                            key=lambda p: p.stat().st_mtime) if log_dir.exists() else []
    rows = parse_cycle_ate(log_candidates[-1]) if log_candidates else []

    # isolate the most recent online_learner.py run
    starts = run_starts(d["gen_id"])
    run_id, lo = len(starts), starts[-1]
    print(f"[viz] {len(starts)} online_learner.py run(s) detected; "
          f"showing latest only (run {run_id}, ticks {lo}:{len(d['gen_id'])})")

    if len(d["swap_events"]) and len(d["t_s"]):
        d["swap_events"] = d["swap_events"][d["swap_events"][:, 0] >= d["t_s"][lo] * 1e9]

    print(d["swap_events"][:])

    for key in ("obs", "output", "truth", "gt_full", "onboard", "gen_id", "t_s"):
        d[key] = d[key][lo:]

    print(f"[viz] obs={d['obs'].shape}  output={d['output'].shape}  "
          f"swap_events={d['swap_events'].shape}")
    print(f"[viz] ground-truth rows={d['truth'].shape[0]}")
    print(f"[viz] online episodes={episodes.shape}")
    print(f"[viz] parsed {len(rows)} cycle ATE result(s) from the run log")

    # odometry + clock alignment
    odo_list = []
    if ULOG_PATH.exists():
        print(f"[viz] loading odometry from {ULOG_PATH}...")
        odo_raw = load_odometry(ULOG_PATH)
        if odo_raw[0] is not None and len(d["t_s"]):
            offset, scale = align_clocks(d["t_s"], d["truth"][:, 2], odo_raw)
            print(f"[viz] clock alignment: offset={offset:.4f}s  scale={scale:.7f}")
            t0 = d["t_s"].min()
            for odo in odo_raw:
                if odo is None:
                    odo_list.append(None)
                else:
                    t, xyz = odo
                    odo_list.append((t * scale + offset - t0, xyz))
        else:
            odo_list = odo_raw
    else:
        odo_list = [None, None]

    if SUMMARY_ONLY:
        summary = summarize_final_generation_error(d, run_id)
        print(f"[viz-summary] {summary}")
        return

    xy_lim, z_lim = compute_global_limits(d, odo_list)

    #plot_inference_trace(d, viz_dir)
    #plot_trajectory_xy(episodes, viz_dir)
    #plot_swap_events(d, viz_dir)
    plot_cycle_ate(rows, viz_dir)
    #plot_current_champion_estimation(d, odo_list, viz_dir, run_id, xy_lim, z_lim)
    #plot_generation_estimation(d, odo_list, viz_dir, run_id, xy_lim, z_lim)

    print(f"[viz] wrote plots to {viz_dir}/")


if __name__ == "__main__":
    main()
