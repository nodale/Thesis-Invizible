#!/usr/bin/env python3
"""Visualize the online_learner.py test-run recorded data in this folder:
  - infer_log.zarr: obs/output (champion's live inference log) + swap_events
  - online_data.zarr: vicon-supervised episodes collected online
  - logs/online_learner_*.log: cur_ATE/new_ATE per cycle (parsed from stdout)

    conda activate alpha311 && python visualize.py

Writes PNGs to nn/viz/. Only the latest online_learner.py run is plotted --
see run_starts() for why runs need to be told apart at all.
"""
import argparse
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import zarr

HERE = Path(__file__).parent
VIZ_DIR = HERE / "viz"
VIZ_DIR.mkdir(exist_ok=True)

NORM = dict(pos=3.0, vel=0.8, angvel=0.5, acc=25.0, thrust=9.81, setpoint=3.0)

# A dwell segment's tail, after its last full lap, is noise unless it's the
# run's still-active segment (then it's the in-progress "current" lap).
PARTIAL_TOLERANCE_S = 2.0
MIN_CURRENT_PARTIAL_S = 1.0


def figure_eight_period_s(default=48.3):
    """One figure-8 lap = 2*pi/EIGHT_OMEGA (same constant online_learner.py uses)."""
    try:
        from alphacore.alpha.providers.trajectory import EIGHT_OMEGA
        return 2 * math.pi / EIGHT_OMEGA
    except Exception:
        return default


def load_infer_log():
    root = zarr.open(str(HERE / "infer_log.zarr"), mode="r")
    obs = np.asarray(root["obs"][:])
    out = np.asarray(root["output"][:])
    swap_events = np.asarray(root["swap_events"][:]) if "swap_events" in root else np.zeros((0, 7))
    # ground_truth/timestamp_ns/generation_id were added after obs/output already
    # had rows -- they only cover the tail. Align by trimming obs/out to match.
    if "ground_truth" in root:
        ground_truth = np.asarray(root["ground_truth"][:])
        timestamp_ns = np.asarray(root["timestamp_ns"][:]).reshape(-1)
        generation_id = np.asarray(root["generation_id"][:]).reshape(-1)
        n_tagged = ground_truth.shape[0]
        obs, out = obs[-n_tagged:], out[-n_tagged:]
    else:
        ground_truth = np.zeros((0, 6))
        timestamp_ns = np.zeros(0, dtype=np.int64)
        generation_id = np.zeros(0, dtype=np.int32)
    return obs, out, swap_events, ground_truth, timestamp_ns, generation_id


def load_online_episodes():
    root = zarr.open(str(HERE / "online_data.zarr"), mode="r")
    return np.asarray(root["episodes"])


def parse_cycle_ate(log_path):
    """Parse '[nn.learner] cur_ATE=X  new_ATE=Y' lines plus swap accept/reject."""
    if not log_path.exists():
        return []
    rows = []
    for m in re.finditer(r"cur_ATE=([\d.]+)\s+new_ATE=([\d.]+)", log_path.read_text()):
        cur, new = float(m.group(1)), float(m.group(2))
        rows.append((cur, new, new < cur))
    return rows


def run_starts(generation_id):
    """Start index of each separate online_learner.py process run.

    generation_id is a per-process counter -- it resets to 0 every time the
    script restarts, so "gen 0" from an earlier run and "gen 0" from a later
    run are unrelated models sharing a label. A drop back to 0 from nonzero
    marks a new run's start; this lets main() show only the latest run
    instead of stitching unrelated runs' generation numbers together."""
    if len(generation_id) == 0:
        return [0]
    resets = np.flatnonzero((generation_id[1:] == 0) & (generation_id[:-1] != 0)) + 1
    return [0, *resets.tolist()]


def plot_inference_trace(obs, n_last=6000):
    obs = obs[-n_last:]
    n = obs.shape[0]
    pos = obs[:, 0:3] * NORM["pos"]
    vel = obs[:, 3:6] * NORM["vel"]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for i, label in enumerate("xyz"):
        axes[0].plot(pos[:, i], label=f"pos {label}", linewidth=0.8)
        axes[1].plot(vel[:, i], label=f"vel {label}", linewidth=0.8)
    axes[0].set_ylabel("position (m)")
    axes[1].set_ylabel("velocity (m/s)")
    axes[1].set_xlabel(f"sample (last {n} of {obs.shape[0]} total, champion inference log)")
    for ax in axes:
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Champion inference log: obs position/velocity (recent window)")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "01_inference_trace.png", dpi=130)
    plt.close(fig)


def plot_trajectory_xy(episodes, n_episodes=20):
    fig, ax = plt.subplots(figsize=(7, 7))
    idxs = np.linspace(0, episodes.shape[0] - 1, min(n_episodes, episodes.shape[0])).astype(int)
    cmap = plt.get_cmap("viridis")
    for i, idx in enumerate(idxs):
        pos = episodes[idx, :, 0:3] * NORM["pos"]
        ax.plot(pos[:, 0], pos[:, 1], color=cmap(i / max(len(idxs) - 1, 1)), alpha=0.7, linewidth=1)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Vicon-supervised episode trajectories (XY), {len(idxs)} of {episodes.shape[0]} episodes shown\n"
                 "(color = episode order, early→late)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "02_episode_trajectories_xy.png", dpi=130)
    plt.close(fig)


def plot_swap_events(swap_events):
    if len(swap_events) == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 7))
    x, y, gen_idx = swap_events[:, 4], swap_events[:, 5], swap_events[:, 2]
    sc = ax.scatter(x, y, c=gen_idx, cmap="plasma", s=120, edgecolor="k", zorder=3)
    for xi, yi, g in zip(x, y, gen_idx):
        ax.annotate(f"gen {int(g)}", (xi, yi), textcoords="offset points", xytext=(8, 8), fontsize=9)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Accepted champion→challenger swap events ({len(swap_events)} total)\n"
                 "position at moment of swap, colored by generation index")
    plt.colorbar(sc, ax=ax, label="generation index")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "03_swap_events.png", dpi=130)
    plt.close(fig)


def plot_cycle_ate(rows):
    if not rows:
        print("[viz] no cur_ATE/new_ATE lines found in the log -- skipping cycle ATE plot")
        return
    cycles = np.arange(len(rows))
    cur = [r[0] for r in rows]
    new = [r[1] for r in rows]
    accepted = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    ax.bar(cycles - width / 2, cur, width, label="champion (current)", color="#888888")
    ax.bar(cycles + width / 2, new, width, label="challenger (new)",
           color=["#2ca02c" if a else "#d62728" for a in accepted])
    for i, a in enumerate(accepted):
        ax.annotate("SWAPPED" if a else "rejected", (i + width / 2, new[i]),
                    textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8,
                    color="#2ca02c" if a else "#d62728")
    ax.set_xticks(cycles)
    ax.set_xticklabels([f"cycle {i}" for i in cycles])
    ax.set_ylabel("ATE (m, lower is better)")
    ax.set_title("Per-cycle champion vs. challenger evaluation (ATE)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(VIZ_DIR / "04_cycle_ate.png", dpi=130)
    plt.close(fig)


def reconstruct_estimated_trajectory(out, ground_truth, pred_dim=6):
    """'What the model itself believed its trajectory was': seed from ground
    truth at the window start, then integrate the model's own predicted
    pos+vel delta (out[:, :pred_dim]) tick by tick. Normalised the same way
    ground_truth is -- caller denormalises for plotting."""
    est = np.zeros_like(ground_truth[:, :pred_dim])
    est[0] = ground_truth[0, :pred_dim]
    for t in range(1, len(out)):
        est[t] = est[t - 1] + out[t - 1, :pred_dim]
    return est


def dwell_segments(generation_id):
    """(gen, lo, hi) for each contiguous run of the same generation_id --
    one champion's unbroken time as the live model, within a single run."""
    if len(generation_id) == 0:
        return []
    edges = np.flatnonzero(np.diff(generation_id)) + 1
    bounds = [0, *edges.tolist(), len(generation_id)]
    return [(int(generation_id[a]), a, b) for a, b in zip(bounds, bounds[1:])]


def lap_segments(generation_id, timestamp_ns, period_s):
    """One entry per figure-8 lap actually flown. A champion that keeps
    winning holds the same generation_id across several laps, so each dwell
    segment is further chopped into consecutive ~period_s windows by
    wall-clock time (the loop's real rate runs under its nominal Hz, so a
    tick-count cutoff would mis-size a lap). A finished segment's leftover
    tail (a few seconds before the swap) isn't a real lap and is dropped --
    except on the last segment, where that tail is the current, still-in-
    progress lap and is worth keeping."""
    laps = []
    segments = dwell_segments(generation_id)
    for seg_i, (gen, seg_lo, seg_hi) in enumerate(segments):
        is_last_segment = seg_i == len(segments) - 1
        lo, round_idx = seg_lo, 0
        while lo < seg_hi - 1:
            elapsed = (timestamp_ns[lo:seg_hi].astype(np.float64) - timestamp_ns[lo]) / 1e9
            hi = lo + max(int(np.searchsorted(elapsed, period_s, side="right")), 1)
            if hi - lo < 2:
                break
            span_s = elapsed[hi - lo - 1]
            partial = span_s < period_s - PARTIAL_TOLERANCE_S
            is_final_window = hi >= seg_hi - 1
            if partial and not (is_last_segment and is_final_window and span_s >= MIN_CURRENT_PARTIAL_S):
                break
            laps.append(dict(gen=gen, round_idx=round_idx, lo=lo, hi=hi, partial=partial))
            lo, round_idx = hi, round_idx + 1
    return laps


def _plot_estimation_window(obs, out, ground_truth, timestamp_ns, lap, period_s, filename, title_prefix):
    """Shared plot body: model's own reconstructed trajectory vs. Vicon
    ground truth, for one lap_segments() entry."""
    lo, hi, partial = lap["lo"], lap["hi"], lap["partial"]
    est = reconstruct_estimated_trajectory(out[lo:hi], ground_truth[lo:hi])
    est_pos = est[:, 0:3] * NORM["pos"]
    gt_pos = ground_truth[lo:hi, 0:3] * NORM["pos"]
    onboard_pos = obs[lo:hi, 0:3] * NORM["pos"]

    duration_s = (timestamp_ns[hi - 1] - timestamp_ns[lo]) / 1e9
    title = f"{title_prefix}gen {lap['gen']} round {lap['round_idx']}"
    tag = " -- PARTIAL" if partial else ""

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].plot(gt_pos[:, 0], gt_pos[:, 1], label="ground truth (vicon)", color="k", linewidth=1.5)
    axes[0].plot(est_pos[:, 0], est_pos[:, 1], label="model's own estimate (reconstructed)",
                 color="#d62728", linewidth=1.2, alpha=0.85)
    axes[0].plot(onboard_pos[:, 0], onboard_pos[:, 1], label="onboard estimator (obs)",
                 color="#1f77b4", linewidth=1, alpha=0.6, linestyle="--")
    axes[0].set_xlabel("x (m)"); axes[0].set_ylabel("y (m)")
    axes[0].set_aspect("equal", adjustable="datalim")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[0].set_title(f"{title}: {duration_s:.1f}s of {period_s:.1f}s ({hi - lo} ticks){tag}")

    err = np.linalg.norm(est_pos - gt_pos, axis=1)
    axes[1].plot(err, color="#d62728")
    axes[1].set_xlabel("tick (this lap)")
    axes[1].set_ylabel("position error (m)")
    axes[1].set_title(f"{title}: |model estimate - ground truth|\nmean={err.mean():.3f}m  max={err.max():.3f}m")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(VIZ_DIR / filename, dpi=130)
    plt.close(fig)
    return err, duration_s


def plot_generation_estimation(obs, out, ground_truth, timestamp_ns, generation_id, run_id):
    """One figure per lap (see lap_segments()) in this run: gen 1 round 0,
    gen 1 round 1, ... (challenger kept losing), gen 2 round 0 (challenger
    finally won), etc."""
    period_s = figure_eight_period_s()
    laps = lap_segments(generation_id, timestamp_ns, period_s)
    if not laps:
        print("[viz] no generation_id data -- skipping per-lap estimation plots")
        return
    for lap_i, lap in enumerate(laps):
        err, duration_s = _plot_estimation_window(
            obs, out, ground_truth, timestamp_ns, lap, period_s,
            filename=f"05_run{run_id}_lap{lap_i:03d}_gen{lap['gen']:03d}_round{lap['round_idx']:02d}_estimation.png",
            title_prefix=f"run {run_id}: ")
        status = "PARTIAL" if lap["partial"] else "full lap"
        print(f"[viz] run {run_id} lap {lap_i} (gen {lap['gen']} round {lap['round_idx']}): "
              f"{duration_s:.1f}s ({status}), mean estimate error {err.mean():.3f}m")


def plot_champion_estimation(obs, out, ground_truth, timestamp_ns, generation_id, run_id):
    """One figure per champion dwell segment (see dwell_segments()) in this
    run, regardless of whether it flew a full figure-8 lap -- every
    generation that ever held champion status gets exactly one plot."""
    period_s = figure_eight_period_s()
    segments = dwell_segments(generation_id)
    if not segments:
        print("[viz] no generation_id data -- skipping per-champion estimation plots")
        return
    for seg_i, (gen, lo, hi) in enumerate(segments):
        lap = dict(gen=gen, round_idx=0, lo=lo, hi=hi, partial=False)
        err, duration_s = _plot_estimation_window(
            obs, out, ground_truth, timestamp_ns, lap, period_s,
            filename=f"05_run{run_id}_champion{seg_i:03d}_gen{gen:03d}_estimation.png",
            title_prefix=f"run {run_id}: ")
        print(f"[viz] run {run_id} champion {seg_i} (gen {gen}): "
              f"{duration_s:.1f}s, mean estimate error {err.mean():.3f}m")


def plot_current_champion_estimation(obs, out, ground_truth, timestamp_ns, generation_id, run_id):
    """Single figure for the CURRENT champion: the most recent lap of this
    run -- "how is the champion doing on the figure-8 it's flying right now"."""
    period_s = figure_eight_period_s()
    laps = lap_segments(generation_id, timestamp_ns, period_s)
    if not laps:
        print("[viz] no generation_id data -- skipping current-champion plot")
        return
    lap = laps[-1]
    err, duration_s = _plot_estimation_window(
        obs, out, ground_truth, timestamp_ns, lap, period_s,
        filename="00_current_champion_estimation.png",
        title_prefix=f"CURRENT champion (run {run_id}) -- ")
    print(f"[viz] current champion (run {run_id}, gen {lap['gen']} round {lap['round_idx']}): "
          f"{duration_s:.1f}s, mean estimate error {err.mean():.3f}m")


def summarize_final_generation_error(obs, out, ground_truth, timestamp_ns, generation_id, run_id):
    """Mean/max position-offset error over the final champion generation's
    laps in this run -- one number to compare "final quality" across two
    separate experiment directories (e.g. reset-on-win vs never-reset)."""
    period_s = figure_eight_period_s()
    laps = [lap for lap in lap_segments(generation_id, timestamp_ns, period_s) if not lap["partial"]]
    if not laps:
        return None
    final_gen = laps[-1]["gen"]
    final_laps = [lap for lap in laps if lap["gen"] == final_gen]
    errs = np.concatenate([
        np.linalg.norm(
            reconstruct_estimated_trajectory(out[lap["lo"]:lap["hi"]], ground_truth[lap["lo"]:lap["hi"]])[:, 0:3] * NORM["pos"]
            - ground_truth[lap["lo"]:lap["hi"], 0:3] * NORM["pos"],
            axis=1)
        for lap in final_laps
    ])
    return dict(run_id=run_id, generation=int(final_gen), n_laps=len(final_laps),
                mean_err=float(errs.mean()), max_err=float(errs.max()))


def main():
    global HERE, VIZ_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE),
                    help="directory containing infer_log.zarr/online_data.zarr/logs "
                         "(default: this script's own folder)")
    ap.add_argument("--summary-only", action="store_true",
                    help="print the final-generation mean/max position error as one line "
                         "and skip generating any PNGs (for comparing two run directories)")
    args = ap.parse_args()
    HERE = Path(args.dir)
    VIZ_DIR = HERE / "viz"
    VIZ_DIR.mkdir(exist_ok=True)

    print(f"[viz] loading recorded data from {HERE}...")
    obs, out, swap_events, ground_truth, timestamp_ns, generation_id = load_infer_log()
    episodes = load_online_episodes()
    log_dir = HERE / "logs"
    log_candidates = sorted(log_dir.glob("online_learner*.log"), key=lambda p: p.stat().st_mtime) if log_dir.exists() else []
    rows = parse_cycle_ate(log_candidates[-1]) if log_candidates else []

    starts = run_starts(generation_id)
    run_id, lo = len(starts), starts[-1]
    print(f"[viz] {len(starts)} online_learner.py run(s) detected in this log; "
          f"showing latest only (run {run_id}, ticks {lo}:{len(generation_id)})")
    if len(swap_events):
        swap_events = swap_events[swap_events[:, 0] >= timestamp_ns[lo]]
    obs, out, ground_truth, timestamp_ns, generation_id = (
        a[lo:] for a in (obs, out, ground_truth, timestamp_ns, generation_id))

    print(f"[viz] infer_log obs={obs.shape} output={out.shape} swap_events={swap_events.shape}")
    print(f"[viz] tagged (ground_truth/timestamp/gen) rows={ground_truth.shape[0]}")
    print(f"[viz] online episodes={episodes.shape}")
    print(f"[viz] parsed {len(rows)} cycle ATE result(s) from the run log")

    if args.summary_only:
        summary = summarize_final_generation_error(obs, out, ground_truth, timestamp_ns, generation_id, run_id)
        print(f"[viz-summary] {summary}")
        return

    plot_inference_trace(obs)
    plot_trajectory_xy(episodes)
    plot_swap_events(swap_events)
    plot_cycle_ate(rows)
    #plot_current_champion_estimation(obs, out, ground_truth, timestamp_ns, generation_id, run_id)
    #plot_champion_estimation(obs, out, ground_truth, timestamp_ns, generation_id, run_id)

    print(f"[viz] wrote plots to {VIZ_DIR}/")


if __name__ == "__main__":
    main()
