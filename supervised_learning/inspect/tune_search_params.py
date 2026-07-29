"""Sweep MCTS search parameters (c_puct x fpu_reduction x sims) against the
Stockfish-annotated suite from build_search_positions.py.

Runs on Kaggle (P100), pasted into one cell — it self-runs. batch_size is fixed
at the value the batch-size experiment settled on (32); this sweep is about move
QUALITY, not speed.

The one trick that makes this cheap
-----------------------------------
MCTS is incremental, so the three-way sweep costs the same as a two-way sweep at
the largest sim count: we run ONE search per (c_puct, fpu, position) out to the
top milestone and read ``get_best_move`` at every milestone along the way. The
move recorded at 256 sims IS exactly the move a 256-sim search would have played
— nothing later in the search changes earlier state, and add_noise is off, so
the whole thing is deterministic. Sims is therefore a free third axis.

Why sims is on the axis and time is not
---------------------------------------
For batch_size, equal wall-clock was the fair test (pure speed knob). c_puct and
fpu barely move throughput, so a fixed time budget would just hand different
cells different sim counts and confound "this c_puct is better" with "this
c_puct happened to run more sims". We fix sims and RECORD elapsed at each
milestone, so the same run can also be read at equal wall-clock if wanted.

Metrics (all from the suite's exact per-move Stockfish cp, side-to-move POV)
---------------------------------------------------------------------------
  * mean cp loss  — headline. Capped at CP_CAP so one hang-into-mate can't own
    the mean. This is the number that tracks playing strength; it only failed to
    resolve the old 400->4000 sims question for lack of samples (n=23), which is
    exactly what the ~400-position suite fixes.
  * blunder rate  — fraction of positions losing >100 / >300 cp. A decomposition
    of the mean, not a rival to it: it shows whether a mean move came from a
    broad quality shift or a few disasters.
  * expected-score loss — cp mapped through 1/(1+10^(-cp/400)) before differ-
    encing, so a 50cp "mistake" in a +600 position (near-zero real cost) stops
    counting like a 50cp mistake in an equal one. Bounded [0,1]; needs no cap.
  * paired vs baseline — for each cell, per-position cp-loss difference against
    the current deployment cell (BASELINE). The shared position difficulty
    cancels, so this resolves far smaller effects than the raw means.

Coarse -> fine
--------------
Run the coarse grid below first. Then edit C_PUCTS / FPUS to a finer grid around
the winner and re-run: results are keyed per (c_puct, fpu) cell and merge into
the same JSON, so only the new cells are searched. Keep MIN_SPREAD and the suite
file unchanged between passes or the position set shifts and pairing breaks (the
script warns if the suite identity changes).

Fairness controls: fresh MCTS per (cell, position) — empty eval cache, no reused
subtree — so no cell inherits a warm cache from an earlier one. Pessimistic vs.
real play (which keeps both warm across a game); the comparison between cells
stays fair.

Usage on Kaggle (Accelerator = GPU P100):
    !pip install chess -q          # cell 1
    # paste this file into cell 2 — it self-runs.

Custom / refinement:
    main(c_pucts=(1.25, 1.5, 1.75), fpus=(0.2, 0.3, 0.4))
"""

import glob
import json
import os
import statistics
import sys
import time


# ---------------------------------------------------------------------------
# Path setup — works locally and when pasted into a Kaggle cell (no __file__).
# ---------------------------------------------------------------------------

def _find_repo_root() -> str:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        cand = os.path.normpath(os.path.join(here, "..", ".."))
        if os.path.isdir(os.path.join(cand, "reinforcement_learning")):
            return cand
    except NameError:
        pass  # pasted into a cell — walk the mounts instead
    for base in ("/kaggle/input", "/kaggle/working", os.getcwd()):
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, _ in os.walk(base):
            if "reinforcement_learning" in dirnames:
                return dirpath
    raise RuntimeError(
        "Could not locate the repo root (folder containing "
        "'reinforcement_learning/')."
    )


REPO_ROOT = _find_repo_root()
RL_DIR = os.path.join(REPO_ROOT, "reinforcement_learning")
for _p in (REPO_ROOT, RL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(REPO_ROOT)  # Converter opens move_lookup.json relative to CWD

import chess
import numpy as np

from helpers.converter import Converter
from monte_carlo_tree_search.mcts_v2 import MCTS
from monte_carlo_tree_search.nodes_and_edges_v2 import Node
from networks.big_network import BigNetwork


# ============================ DEFAULT GRID / CONFIG =============================
# The deployment setting, and the baseline every cell is paired against. MUST be
# a point on the grid or pairing is skipped.
BASELINE_C_PUCT = 1.5
BASELINE_FPU = 0.3

C_PUCTS = (0.75, 1.5, 2.25, 3.0, 4.0)
FPUS = (None, 0.0, 0.15, 0.3, 0.5)      # None = legacy unvisited-Q=0 (distinct from 0.0)
MILESTONES = (32, 64, 128, 256, 512, 1024)  # snapshot sims; must be multiples of BATCH_SIZE
BATCH_SIZE = 32                          # fixed by the batch-size experiment

MIN_SPREAD = 50   # drop positions where best_cp - worst_cp < this (nothing to
#                   get wrong: every legal move is within MIN_SPREAD of best).
CP_CAP = 1000     # cap on per-move cp loss for the MEAN (blunder flags use raw loss)

OUT_JSON = os.path.join("/kaggle/working", "search_sweep_results.json") \
    if os.path.isdir("/kaggle/working") \
    else os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals()
                      else ".", "search_sweep_results.json")
# ===============================================================================


def expected_score(cp: float) -> float:
    """Logistic expected game score in [0,1] from a centipawn eval (mover POV)."""
    return 1.0 / (1.0 + 10.0 ** (-cp / 400.0))


def _find_suite(explicit: str | None) -> str:
    if explicit:
        if os.path.exists(explicit):
            return explicit
        raise FileNotFoundError(explicit)
    patterns = [
        "/kaggle/input/**/search_positions.json",
        os.path.join(REPO_ROOT, "supervised_learning", "inspect", "search_positions.json"),
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[-1]
    raise FileNotFoundError(
        "search_positions.json not found. Build it with "
        "build_search_positions.py and upload it as a Kaggle dataset, or pass "
        "suite_path=..."
    )


def _find_weights(explicit: str | None) -> str:
    if explicit:
        if os.path.exists(explicit):
            return explicit
        raise FileNotFoundError(explicit)
    patterns = [
        "/kaggle/working/checkpoints/sl_best.weights.h5",
        "/kaggle/working/checkpoints/sl_*.weights.h5",
        "/kaggle/input/**/sl_*.weights.h5",
        os.path.join(REPO_ROOT, "supervised_learning", "checkpoints", "sl_*.weights.h5"),
        os.path.join(REPO_ROOT, "*.weights.h5"),
    ]
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[-1]
    raise FileNotFoundError("No sl_*.weights.h5 found. Pass weights_path=...")


def _report_device() -> None:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("!! No GPU visible - timings will not transfer to a GPU run.")
        return
    print(f"GPUs visible to TensorFlow: {len(gpus)}")
    for g in gpus:
        try:
            det = tf.config.experimental.get_device_details(g)
            print(f"  {g.name}  {det.get('device_name', '?')}")
        except Exception:
            print(f"  {g.name}")
    if len(gpus) > 1:
        print("  NOTE: TF uses GPU:0 only. A single P100 beats dual T4 here.")


# ---------------------------------------------------------------------------
# Suite loading + scoring
# ---------------------------------------------------------------------------

def load_suite(path: str, min_spread: int) -> tuple[dict, list[dict], dict, int]:
    """Load the annotated suite and drop 'nothing to get wrong' positions.

    Returns (suite_meta, kept_positions, index_by_fen, dropped_count).
    index_by_fen[fen] = {"moves": [(uci,cp),...] best-first, "phase", "n_legal",
                         "best_cp", "cp": {uci: cp}, "rank": {uci: 1-based}}.
    """
    with open(path) as f:
        blob = json.load(f)
    meta = blob["meta"]
    kept: list[dict] = []
    index: dict[str, dict] = {}
    dropped = 0
    for p in blob["positions"]:
        moves = [(uci, int(cp)) for uci, cp in p["moves"]]
        best_cp, worst_cp = moves[0][1], moves[-1][1]
        if best_cp - worst_cp < min_spread:
            dropped += 1
            continue
        kept.append(p)
        index[p["fen"]] = {
            "moves": moves,
            "phase": p.get("phase", "?"),
            "n_legal": len(moves),
            "best_cp": best_cp,
            "cp": {uci: cp for uci, cp in moves},
            "rank": {uci: i + 1 for i, (uci, _) in enumerate(moves)},
        }
    return meta, kept, index, dropped


def score_move(annot: dict, uci: str, cp_cap: int) -> dict:
    """Score a chosen move against a position's annotation.

    A miss means the chosen move is not among the position's legal moves — a
    bug (encoding / stale suite), surfaced loudly rather than silently scored 0.
    """
    if uci not in annot["rank"]:
        raise KeyError(
            f"move {uci} not in annotation for a position with "
            f"{annot['n_legal']} legal moves — suite stale or move encoding "
            f"mismatch."
        )
    best_cp = annot["best_cp"]
    chosen_cp = annot["cp"][uci]
    raw_loss = best_cp - chosen_cp
    esc_loss = expected_score(best_cp) - expected_score(chosen_cp)
    return {
        "rank": annot["rank"][uci],
        "cp_loss_raw": raw_loss,
        "cp_loss": min(raw_loss, cp_cap),
        "esc_loss": esc_loss,
        "n_legal": annot["n_legal"],
    }


# ---------------------------------------------------------------------------
# Search with per-milestone snapshots
# ---------------------------------------------------------------------------

def search_with_snapshots(
    mcts: MCTS, root: Node, batch_size: int, milestones: tuple[int, ...]
) -> dict[int, dict]:
    """Run one batched search out to max(milestones), snapshotting the chosen
    move / elapsed / top-visit share at each milestone.

    Driven one batch-round at a time (num_simulations = batch_size per call) so
    every milestone lands between rounds, in a clean tree state (virtual losses
    are all reverted inside each search_batched call). add_noise is off, so the
    move at each milestone equals what a search of that many sims would pick.
    """
    mcts.num_simulations = batch_size
    snaps: dict[int, dict] = {}
    sims = 0
    t0 = time.perf_counter()
    for m in sorted(milestones):
        while sims < m and not root.is_terminal:
            mcts.search_batched(root, add_noise=False, batch_size=batch_size)
            sims += batch_size
        elapsed = time.perf_counter() - t0
        best = mcts.get_best_move(root, temperature=0)
        top_n = max((e.N for e in root.edges), default=0)
        snaps[m] = {
            "move": best.uci(),
            "elapsed": elapsed,
            "sims": sims,
            "topvis": top_n / max(1, root.total_visits),
        }
        if root.is_terminal:  # search can't grow further; freeze remaining
            for mm in sorted(milestones):
                snaps.setdefault(mm, snaps[m])
            break
    return snaps


# ---------------------------------------------------------------------------
# Sweep driver (resumable)
# ---------------------------------------------------------------------------

def cell_key(c_puct: float, fpu: float | None) -> str:
    return f"cpuct={c_puct:.2f},fpu={'None' if fpu is None else f'{fpu:.2f}'}"


def _suite_id(meta: dict) -> dict:
    """Identity fields that must match across coarse/fine passes for pairing."""
    return {k: meta.get(k) for k in ("created", "suite_size", "sf_depth", "seed")}


def run_sweep(
    net: BigNetwork,
    converter: Converter,
    suite: list[dict],
    suite_meta: dict,
    c_pucts: tuple[float, ...],
    fpus: tuple,
    milestones: tuple[int, ...],
    batch_size: int,
    cp_cap: int,
    min_spread: int,
    out_path: str,
    force: bool,
) -> dict:
    """Search every (c_puct, fpu) cell over the suite. Save per cell; skip cells
    already present (unless force). Returns the results dict."""
    results = _load_results(out_path)
    want_meta = {
        "batch_size": batch_size,
        "milestones": list(milestones),
        "min_spread": min_spread,
        "cp_cap": cp_cap,
        "suite_id": _suite_id(suite_meta),
        "baseline": cell_key(BASELINE_C_PUCT, BASELINE_FPU),
        "weights": getattr(net, "_loaded_weights_name", "?"),
    }
    if results.get("meta") and not force:
        _check_meta_compat(results["meta"], want_meta)
    else:
        results = {"meta": want_meta, "cells": {}}
    cells = results["cells"]

    grid = [(c, f) for c in c_pucts for f in fpus]
    todo = [(c, f) for c, f in grid if force or cell_key(c, f) not in cells]
    print(f"\ngrid: {len(grid)} cells ({len(c_pucts)} c_puct x {len(fpus)} fpu), "
          f"{len(todo)} to run, {len(grid) - len(todo)} cached")
    print(f"suite: {len(suite)} positions, up to {max(milestones)} sims each\n")

    done_times: list[float] = []
    for ci, (c_puct, fpu) in enumerate(grid, 1):
        key = cell_key(c_puct, fpu)
        if key in cells and not force:
            continue
        rows = []
        t_cell = time.perf_counter()
        n_err = 0
        for p in suite:
            fen = p["fen"]
            try:
                mcts = MCTS(
                    network=net, converter=converter,
                    num_simulations=batch_size, c_puct=c_puct, fpu_reduction=fpu,
                )
                root = Node(chess.Board(fen))
                snaps = search_with_snapshots(mcts, root, batch_size, milestones)
                rows.append({"fen": fen, "snaps": snaps})
            except Exception as e:  # one bad position must not kill a 2h run
                n_err += 1
                if n_err <= 3:
                    print(f"    !! {key} skipped a position ({fen}): {e}")
                continue
        dt = time.perf_counter() - t_cell
        cells[key] = {"c_puct": c_puct, "fpu": fpu, "rows": rows, "wall": dt}
        _save_results(out_path, results)
        done_times.append(dt)
        mean_dt = sum(done_times) / len(done_times)
        remaining = mean_dt * (len(todo) - len(done_times))
        print(f"[cell {len(done_times)}/{len(todo)}] {key:<24} {len(rows)} pos "
              f"in {dt:6.1f}s{f'  ({n_err} skipped)' if n_err else ''}"
              f"   ~{remaining/60:.0f}m left")
    return results


def _load_results(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_results(path: str, results: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f)
    os.replace(tmp, path)


def _check_meta_compat(have: dict, want: dict) -> None:
    for k in ("batch_size", "milestones", "min_spread"):
        if have.get(k) != want.get(k):
            print(f"!! results {k}={have.get(k)} != current {want.get(k)}; "
                  f"mixing incompatible runs. Delete {OUT_JSON} or set force=True.")
    if have.get("suite_id") != want.get("suite_id"):
        print("!! the suite JSON identity changed since these results were "
              "started - positions may differ, pairing is unreliable. Rebuild "
              "from one suite or delete the results file.")


# ---------------------------------------------------------------------------
# Summary / analysis (recomputed from saved moves — no re-search needed)
# ---------------------------------------------------------------------------

def _cell_metrics(rows, index, milestone, cp_cap):
    """Aggregate metrics for one cell at one milestone. Positions missing from
    the current suite index (e.g. re-thresholded out) are skipped."""
    cp, esc, ranks, topvis, elapsed = [], [], [], [], []
    blun100 = blun300 = top1 = top3 = 0
    per_fen = {}
    for r in rows:
        annot = index.get(r["fen"])
        snap = r["snaps"].get(str(milestone)) or r["snaps"].get(milestone)
        if annot is None or snap is None:
            continue
        s = score_move(annot, snap["move"], cp_cap)
        cp.append(s["cp_loss"]); esc.append(s["esc_loss"]); ranks.append(s["rank"])
        topvis.append(snap["topvis"]); elapsed.append(snap["elapsed"])
        blun100 += s["cp_loss_raw"] > 100
        blun300 += s["cp_loss_raw"] > 300
        top1 += s["rank"] == 1
        top3 += s["rank"] <= 3
        per_fen[r["fen"]] = s["cp_loss"]
    n = len(cp)
    if n == 0:
        return None
    return {
        "n": n,
        "mean_cp": sum(cp) / n,
        "med_cp": statistics.median(cp),
        "esc_loss": sum(esc) / n * 100,      # in "expected-score %" units
        "blun100": blun100 / n * 100,
        "blun300": blun300 / n * 100,
        "top1": top1 / n * 100,
        "top3": top3 / n * 100,
        "mean_rank": sum(ranks) / n,
        "topvis": sum(topvis) / n * 100,
        "elapsed": sum(elapsed) / n,
        "per_fen": per_fen,
    }


def summarize(results: dict, index: dict, milestones: tuple[int, ...],
              cp_cap: int, report_milestone: int | None = None) -> None:
    cells = results["cells"]
    keys = list(cells.keys())
    if not keys:
        print("no cells to summarize.")
        return
    report_milestone = report_milestone or max(milestones)
    baseline_key = results["meta"].get("baseline")

    # metrics[key][milestone] = dict
    metrics = {
        k: {m: _cell_metrics(cells[k]["rows"], index, m, cp_cap) for m in milestones}
        for k in keys
    }

    # ---- Table 1: mean cp grid, cells x milestones (the core result) ----
    print("\n" + "=" * 92)
    print(f"MEAN CP LOSS  (cap {cp_cap})  - rows = (c_puct, fpu) cell, cols = sims")
    print("lower is better. '*' marks the current deployment cell.")
    print("=" * 92)
    hdr = f"{'cell':<24}" + "".join(f"{m:>9}" for m in milestones) + f"{'best@':>8}"
    print(hdr); print("-" * len(hdr))
    # order cells by their best (min over milestones) mean cp
    def cell_best(k):
        vals = [metrics[k][m]["mean_cp"] for m in milestones if metrics[k][m]]
        return min(vals) if vals else float("inf")
    for k in sorted(keys, key=cell_best):
        star = " *" if k == baseline_key else "  "
        cells_row = ""
        best_m, best_v = None, float("inf")
        for m in milestones:
            mv = metrics[k][m]
            if mv is None:
                cells_row += f"{'-':>9}"; continue
            cells_row += f"{mv['mean_cp']:>9.1f}"
            if mv["mean_cp"] < best_v:
                best_v, best_m = mv["mean_cp"], m
        print(f"{k+star:<24}{cells_row}{best_m if best_m else '-':>8}")

    # ---- Table 2: full metric breakdown at the report milestone ----
    print("\n" + "=" * 100)
    print(f"DETAIL @ {report_milestone} sims  (n = scored positions)")
    print("=" * 100)
    cols = (f"{'cell':<24}{'n':>4}{'mean cp':>8}{'med':>5}{'esc%':>6}"
            f"{'bl>100':>7}{'bl>300':>7}{'top1':>6}{'top3':>6}{'rank':>6}"
            f"{'tvis%':>6}{'s/mv':>6}")
    print(cols); print("-" * len(cols))
    order = sorted(keys, key=lambda k: (metrics[k][report_milestone]["mean_cp"]
                                        if metrics[k][report_milestone] else 1e9))
    for k in order:
        mv = metrics[k][report_milestone]
        if mv is None:
            continue
        star = " *" if k == baseline_key else "  "
        print(f"{k+star:<24}{mv['n']:>4}{mv['mean_cp']:>8.1f}{mv['med_cp']:>5.0f}"
              f"{mv['esc_loss']:>6.2f}{mv['blun100']:>6.1f}%{mv['blun300']:>6.1f}%"
              f"{mv['top1']:>5.0f}%{mv['top3']:>5.0f}%{mv['mean_rank']:>6.1f}"
              f"{mv['topvis']:>6.1f}{mv['elapsed']:>6.2f}")

    # ---- Table 3: paired vs baseline at the report milestone ----
    if baseline_key in metrics and metrics[baseline_key][report_milestone]:
        base_fen = metrics[baseline_key][report_milestone]["per_fen"]
        print("\n" + "=" * 92)
        print(f"PAIRED vs BASELINE ({baseline_key}) @ {report_milestone} sims")
        print("mean d_cp < 0 means the cell loses LESS cp than baseline "
              "(better). better/worse count positions.")
        print("=" * 92)
        ph = f"{'cell':<24}{'npair':>6}{'mean d_cp':>11}{'better':>8}{'worse':>8}{'tie':>6}"
        print(ph); print("-" * len(ph))
        pair_rows = []
        for k in keys:
            if k == baseline_key:
                continue
            mv = metrics[k][report_milestone]
            if mv is None:
                continue
            deltas = []
            better = worse = tie = 0
            for fen, loss in mv["per_fen"].items():
                if fen in base_fen:
                    d = loss - base_fen[fen]
                    deltas.append(d)
                    if d < 0: better += 1
                    elif d > 0: worse += 1
                    else: tie += 1
            if deltas:
                pair_rows.append((sum(deltas) / len(deltas), k, len(deltas),
                                  better, worse, tie))
        for mean_d, k, npair, better, worse, tie in sorted(pair_rows):
            print(f"{k:<24}{npair:>6}{mean_d:>+10.1f}{better:>8}{worse:>8}{tie:>6}")

    # ---- Table 4: sims trend for the best cell (is more sims helping?) ----
    best_cell = min(keys, key=cell_best)
    print("\n" + "=" * 60)
    print(f"SIMS TREND for best cell {best_cell} (mean cp per milestone)")
    print("=" * 60)
    for m in milestones:
        mv = metrics[best_cell][m]
        if mv:
            print(f"  {m:>5} sims  mean cp {mv['mean_cp']:>7.1f}   "
                  f"bl>300 {mv['blun300']:>4.1f}%   s/mv {mv['elapsed']:.2f}")
    vals = [(m, metrics[best_cell][m]["mean_cp"]) for m in milestones
            if metrics[best_cell][m]]
    if len(vals) >= 2:
        argmin_m = min(vals, key=lambda x: x[1])[0]
        last_m = vals[-1][0]
        if argmin_m != last_m:
            print(f"\n  NOTE: mean cp is lowest at {argmin_m} sims, not at "
                  f"{last_m} - here, more search does not help (and may hurt). "
                  f"One thing that produces this is the value head backing up "
                  f"worse estimates than the policy prior as N grows (PUCT "
                  f"leans on Q more as visits accumulate), but this suite can't "
                  f"prove the cause. If it holds across cells, a visit-scaled "
                  f"c_puct is the natural follow-up; confirm against a real "
                  f"MCTS match before acting.")

    print("\nReading guide: Table 1 localizes the basin (where to refine next); "
          "Table 3 is the highest-power test of whether a cell beats today's "
          "settings; Table 4 answers your 'do more sims hurt?' question directly.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(
    c_pucts: tuple[float, ...] = C_PUCTS,
    fpus: tuple = FPUS,
    milestones: tuple[int, ...] = MILESTONES,
    batch_size: int = BATCH_SIZE,
    min_spread: int = MIN_SPREAD,
    cp_cap: int = CP_CAP,
    weights_path: str | None = None,
    suite_path: str | None = None,
    out_path: str = OUT_JSON,
    report_milestone: int | None = None,
    force: bool = False,
) -> dict:
    bad = [m for m in milestones if m % batch_size != 0]
    if bad:
        raise ValueError(f"milestones {bad} are not multiples of batch_size "
                         f"{batch_size}; snapshots would overshoot.")

    _report_device()

    suite_file = _find_suite(suite_path)
    suite_meta, suite, index, dropped = load_suite(suite_file, min_spread)
    print(f"Suite: {os.path.basename(suite_file)}  "
          f"(Stockfish depth {suite_meta.get('sf_depth', '?')})")
    print(f"  {len(suite)} positions kept, {dropped} dropped "
          f"(best-worst cp spread < {min_spread})")

    wpath = _find_weights(weights_path)
    net = BigNetwork()
    net.load(wpath)
    net._loaded_weights_name = os.path.basename(wpath)
    print(f"Weights: {os.path.basename(wpath)}")
    converter = Converter()

    # Trace the tf.function before timing (first call builds the graph).
    net.predict_batch(np.zeros((1, 8, 8, 20), dtype=np.float32))

    n_run = len([1 for c in c_pucts for f in fpus])
    print(f"\nRough estimate: {n_run} cells x {len(suite)} pos x "
          f"~{max(milestones)} sims. Watch the first cell's time, then "
          f"multiply. Resumable per cell -> a timeout costs at most one cell.")

    results = run_sweep(
        net, converter, suite, suite_meta, c_pucts, fpus, milestones,
        batch_size, cp_cap, min_spread, out_path, force,
    )
    summarize(results, index, milestones, cp_cap, report_milestone)
    print(f"\nResults JSON: {out_path}")
    return results


if __name__ == "__main__":
    main()
