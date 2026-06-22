"""
Assemble game + puzzle + tablebase manifests into balanced training chunks.

Run from the chess_bot root (after the three extractors have produced pools/):
    python supervised_learning/create_dataset/process_game_data.py
    python supervised_learning/create_dataset/process_puzzle_data.py
    python supervised_learning/create_dataset/process_tablebase_data.py
    python supervised_learning/create_dataset/build_dataset.py

Every output chunk (50k positions) is simultaneously:
    * source-mixed  85% games / 10% puzzles / 5% tablebase, and
    * WDL-balanced  ~1/3 win / 1/3 draw / 1/3 loss   (the value head was
      collapsing to "draw" on the draw-heavy raw games).

Because puzzles are all wins and tablebase walks are wins+losses (no draws),
the per-chunk source x WDL split isn't free: the only spare freedom is how the
2.5% tablebase slice divides between win and loss. We pick that split to
MAXIMISE the number of balanced chunks the available pools support (the
draw-heavy games make win/loss the binding classes; puzzles refill win and
tablebase refills loss, so more of the abundant draws survive).

Output: supervised_learning/processed_data/chunk_XXXX.npz  (boards/policies/values),
the exact format train_supervised.py consumes.
"""

import glob
import multiprocessing
import os
import random
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_common as dc  # noqa: E402

POOL_DIR  = os.path.join(dc.repo_root(), "supervised_learning", "pools")
OUT_DIR   = os.path.join(dc.repo_root(), "supervised_learning", "processed_data")
BUILD_DIR = os.path.join(OUT_DIR, "_build")

CHUNK_SIZE = 50_000
# Target source mix per chunk (must sum to CHUNK_SIZE).
SRC_GAME, SRC_PUZZLE, SRC_TB = 42_500, 5_000, 2_500
# Target WDL mix per chunk (must sum to CHUNK_SIZE).
COL_WIN, COL_DRAW, COL_LOSS = 16_667, 16_667, 16_666

MAX_CHUNKS = None         # cap output chunks (None = as many as the pools allow)
ASSEMBLE_WORKERS = 8      # parallel encoders; each holds ~0.5 GB while writing a chunk
SEED = 7

SOURCES = ("game", "puzzle", "tablebase")


def _source_of(path: str) -> str:
    b = os.path.basename(path)
    if b.startswith("pool_games"):     return "game"
    if b.startswith("pool_puzzles"):   return "puzzle"
    if b.startswith("pool_tablebase"): return "tablebase"
    raise ValueError(f"unknown pool file: {b}")


def _pool_files() -> list[str]:
    files = sorted(glob.glob(os.path.join(POOL_DIR, "pool_*.csv")))
    if not files:
        print(f"ERROR: no pool_*.csv in {POOL_DIR}. Run the three extractors first.")
        sys.exit(1)
    return files


# ---------------------------------------------------------------------------
# Pass A — count positions per (source, wdl)
# ---------------------------------------------------------------------------

def count_groups(files: list[str]) -> dict[tuple[str, int], int]:
    counts: dict[tuple[str, int], int] = {}
    for path in files:
        src = _source_of(path)
        with open(path, encoding="utf-8") as f:
            for line in f:
                w = int(line[line.rfind(",") + 1:])
                counts[(src, w)] = counts.get((src, w), 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Plan — choose the tablebase win/loss split that maximises the chunk count,
# and derive the per-chunk quota for every (source, wdl) group.
# ---------------------------------------------------------------------------

def plan(counts: dict[tuple[str, int], int]):
    def c(s, w): return counts.get((s, w), 0)

    best = None  # (N, tb_win, quotas)
    for tb_win in range(0, SRC_TB + 1):
        tb_loss = SRC_TB - tb_win
        q = {
            ("puzzle", dc.WDL_WIN):    SRC_PUZZLE,           # puzzles are all wins
            ("tablebase", dc.WDL_WIN): tb_win,
            ("tablebase", dc.WDL_LOSS): tb_loss,
            ("game", dc.WDL_WIN):  COL_WIN  - SRC_PUZZLE - tb_win,
            ("game", dc.WDL_DRAW): COL_DRAW,                  # all draws come from games
            ("game", dc.WDL_LOSS): COL_LOSS - tb_loss,
        }
        if any(v < 0 for v in q.values()):
            continue
        # N is bounded by the scarcest pool relative to its per-chunk demand.
        N = min((c(s, w) // per) for (s, w), per in q.items() if per > 0)
        if best is None or N > best[0]:
            best = (N, tb_win, q)

    N, tb_win, q = best
    if MAX_CHUNKS is not None:
        N = min(N, MAX_CHUNKS)
    return N, tb_win, q


# ---------------------------------------------------------------------------
# Pass B — select a random subset of each group and route rows into per-chunk
# manifest files. Contiguous assignment (rank // quota) keeps only a handful of
# chunk files active at once, so this streams in tiny memory.
# ---------------------------------------------------------------------------

class _HandleCache:
    """Append-mode file handles, LRU-evicted so we never hold too many open."""
    def __init__(self, limit: int = 64):
        self.limit = limit
        self.handles: dict[int, object] = {}

    def write(self, chunk_id: int, line: str) -> None:
        h = self.handles.get(chunk_id)
        if h is None:
            if len(self.handles) >= self.limit:
                old_id, old_h = next(iter(self.handles.items()))
                old_h.close()
                del self.handles[old_id]
            h = open(os.path.join(BUILD_DIR, f"c{chunk_id:04d}.txt"), "a", encoding="utf-8")
            self.handles[chunk_id] = h
        h.write(line)

    def close(self) -> None:
        for h in self.handles.values():
            h.close()
        self.handles.clear()


def route(files: list[str], counts, N: int, q: dict) -> None:
    if os.path.isdir(BUILD_DIR):
        shutil.rmtree(BUILD_DIR)
    os.makedirs(BUILD_DIR)

    rng = np.random.default_rng(SEED)
    # For each group: which of its (in-stream-order) occurrences to keep, sorted.
    kept_idx, kept_ptr = {}, {}
    for (s, w), per in q.items():
        if per <= 0:
            continue
        keep = N * per
        total = counts[(s, w)]
        sel = rng.choice(total, size=keep, replace=False)
        sel.sort()
        kept_idx[(s, w)] = sel
        kept_ptr[(s, w)] = 0

    occ = {k: 0 for k in kept_idx}     # per-group occurrence counter
    rank = {k: 0 for k in kept_idx}    # per-group rank among kept rows
    cache = _HandleCache()

    for path in files:
        src = _source_of(path)
        with open(path, encoding="utf-8") as f:
            for line in f:
                w = int(line[line.rfind(",") + 1:])
                key = (src, w)
                sel = kept_idx.get(key)
                if sel is None:
                    continue
                ptr = kept_ptr[key]
                i = occ[key]
                occ[key] = i + 1
                if ptr < len(sel) and sel[ptr] == i:
                    r = rank[key]
                    rank[key] = r + 1
                    kept_ptr[key] = ptr + 1
                    target = r // q[key]          # contiguous -> few active files
                    cache.write(target, line)
    cache.close()


# ---------------------------------------------------------------------------
# Pass C — encode each per-chunk manifest into the npz format (parallel).
# ---------------------------------------------------------------------------

def encode_chunk(args) -> tuple[str, int]:
    chunk_path, seed = args
    with open(chunk_path, encoding="utf-8") as f:
        rows = [dc.parse_manifest_line(l) for l in f]
    random.Random(seed).shuffle(rows)          # shuffle sources/classes within the chunk

    n = len(rows)
    boards   = np.empty((n, 8, 8, 20), dtype=np.float16)
    policies = np.zeros((n, dc.POLICY_SIZE), dtype=np.float32)
    values   = np.zeros((n, dc.N_WDL),       dtype=np.float32)
    for i, (fen, idx, wdl) in enumerate(rows):
        b, p, v = dc.encode_sample(fen, idx, wdl)
        boards[i] = b
        policies[i] = p
        values[i] = v

    cid = os.path.splitext(os.path.basename(chunk_path))[0][1:]  # "c0007" -> "0007"
    out = os.path.join(OUT_DIR, f"chunk_{cid}.npz")
    np.savez_compressed(out, boards=boards, policies=policies, values=values)
    return out, n


def assemble() -> None:
    chunk_files = sorted(glob.glob(os.path.join(BUILD_DIR, "c*.txt")))
    work = [(p, SEED + i) for i, p in enumerate(chunk_files)]
    with multiprocessing.Pool(processes=min(ASSEMBLE_WORKERS, len(work))) as pool:
        for out, n in pool.imap_unordered(encode_chunk, work):
            print(f"  wrote {os.path.basename(out)}  ({n:,} positions)", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    files = _pool_files()

    print("=== Dataset Assembler ===")
    print(f"Pools : {POOL_DIR}")
    print(f"Out   : {OUT_DIR}/chunk_*.npz\n")

    t0 = time.time()
    print("Pass A: counting pools ...", flush=True)
    counts = count_groups(files)
    for s in SOURCES:
        row = {w: counts.get((s, w), 0) for w in (0, 1, 2)}
        tot = sum(row.values())
        print(f"  {s:10s} win {row[0]:>10,}  draw {row[1]:>10,}  loss {row[2]:>10,}  | {tot:>11,}")

    N, tb_win, q = plan(counts)
    if N <= 0:
        print("\nERROR: pools too small / imbalanced to build even one balanced chunk.")
        sys.exit(1)
    print(f"\nPlan: {N} chunks of {CHUNK_SIZE:,}  (tablebase split {tb_win} win / {SRC_TB - tb_win} loss per chunk)")
    print("  per-chunk quota:")
    for (s, w), per in sorted(q.items()):
        if per > 0:
            print(f"    {s:10s} {['win','draw','loss'][w]:4s}: {per:,}")
    print(f"  totals/chunk: game {SRC_GAME:,} | puzzle {SRC_PUZZLE:,} | tb {SRC_TB:,}"
          f"   ||  win {COL_WIN:,} | draw {COL_DRAW:,} | loss {COL_LOSS:,}")

    print(f"\nPass B: routing rows into {N} per-chunk manifests ...", flush=True)
    route(files, counts, N, q)

    print(f"Pass C: encoding chunks ({ASSEMBLE_WORKERS} workers) ...", flush=True)
    assemble()

    shutil.rmtree(BUILD_DIR, ignore_errors=True)
    print(f"\n=== Done in {time.time() - t0:.0f}s — {N} chunks -> {OUT_DIR}/ ===")


if __name__ == "__main__":
    main()
