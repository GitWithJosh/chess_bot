"""
Assemble game + puzzle + tablebase manifests into balanced training chunks.

Run from the chess_bot root (after the three extractors have produced pools/):
    python supervised_learning/create_dataset/process_game_data.py
    python supervised_learning/create_dataset/process_puzzle_data.py
    python supervised_learning/create_dataset/process_tablebase_data.py
    python supervised_learning/create_dataset/build_dataset.py

To build with Stockfish-blended (soft) value targets, split the run so the
annotator can sit between routing and encoding:

    python supervised_learning/create_dataset/build_dataset.py --route-only
    python supervised_learning/create_dataset/annotate_stockfish.py    # overnight
    python supervised_learning/create_dataset/build_dataset.py --encode-only

Balancing deliberately still runs on the GAME-OUTCOME label, so the selected
rows — and therefore the chunk count — are identical either way; Stockfish only
changes the value target inside each row (dataset_common.blend_value). Encoding
manifests that were never annotated is still valid and reproduces the old
one-hot targets exactly.

Every output chunk (50k positions) is simultaneously:
    * source-mixed  85% games / 10% puzzles / 5% tablebase, and
    * WDL-balanced  ~1/3 win / 1/3 draw / 1/3 loss   (the value head was
      collapsing to "draw" on the draw-heavy raw games).

The within-source WDL splits are FIXED (2026-07 anti-shortcut experiment:
puzzles now carry defender-side loss rows and tablebase carries drawn rows,
so no source is a single-class giveaway the value head can exploit):

              win     draw    loss   | total
    game     13,334  15,834  13,332  | 42,500
    puzzle    2,500       0   2,500  |  5,000
    tablebase   833     833     834  |  2,500
    total    16,667  16,667  16,666  | 50,000

The chunk count is whatever the scarcest (source, wdl) pool supports.

Output: supervised_learning/processed_data/chunk_XXXX.npz  (boards/policies/
values/sources), the exact format train_supervised.py consumes. The sources
array (uint8, see dataset_common.SOURCE_IDS) tags every position with its data
source, enabling per-source val metrics and per-source loss weighting.
"""

import argparse
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
# Fixed within-source WDL splits (see module docstring); games fill the rest.
PUZ_WIN, PUZ_LOSS = 2_500, 2_500              # puzzles: solver wins / defender losses
TB_WIN, TB_DRAW, TB_LOSS = 833, 833, 834      # tablebase: ~1/3 each

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
# Plan — fixed per-(source, wdl) quotas; the chunk count is whatever the
# scarcest pool supports.
# ---------------------------------------------------------------------------

def plan(counts: dict[tuple[str, int], int]):
    q = {
        ("puzzle", dc.WDL_WIN):     PUZ_WIN,
        ("puzzle", dc.WDL_LOSS):    PUZ_LOSS,
        ("tablebase", dc.WDL_WIN):  TB_WIN,
        ("tablebase", dc.WDL_DRAW): TB_DRAW,
        ("tablebase", dc.WDL_LOSS): TB_LOSS,
        ("game", dc.WDL_WIN):  COL_WIN  - PUZ_WIN  - TB_WIN,
        ("game", dc.WDL_DRAW): COL_DRAW - TB_DRAW,
        ("game", dc.WDL_LOSS): COL_LOSS - PUZ_LOSS - TB_LOSS,
    }
    assert sum(q.values()) == CHUNK_SIZE and all(v >= 0 for v in q.values())
    assert PUZ_WIN + PUZ_LOSS == SRC_PUZZLE
    assert TB_WIN + TB_DRAW + TB_LOSS == SRC_TB

    # N is bounded by the scarcest pool relative to its per-chunk demand.
    N = min(counts.get(k, 0) // per for k, per in q.items() if per > 0)
    if MAX_CHUNKS is not None:
        N = min(N, MAX_CHUNKS)
    return N, q


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
        src_id = dc.SOURCE_IDS[src]
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
                    # Routed rows carry the source id as a 4th field, so the
                    # encoder can emit the per-sample sources array.
                    cache.write(target, f"{line.rstrip(chr(10))},{src_id}\n")
    cache.close()


# ---------------------------------------------------------------------------
# Pass C — encode each per-chunk manifest into the npz format (parallel).
# ---------------------------------------------------------------------------

def _parse_routed_line(line: str):
    """fen,move_idx,wdl,source_id[,w,d,l] -> (fen, move_idx, wdl, source_id, sf_wdl).

    The trailing w,d,l are added by annotate_stockfish.py --join; -1,-1,-1 marks
    a row with no annotation (tablebase, or a position Stockfish never reached).
    Manifests that were never joined parse fine and yield sf_wdl None, so the
    outcome-only build still works unchanged.
    """
    parts = line.rstrip("\n").rsplit(",", 6)
    if len(parts) == 7:
        fen, mi, w, s, a, b, c = parts
        sf = None if a == "-1" else (int(a), int(b), int(c))
        return fen, int(mi), int(w), int(s), sf
    fen, mi, w, s = line.rstrip("\n").rsplit(",", 3)
    return fen, int(mi), int(w), int(s), None


def encode_chunk(args) -> tuple[str, int, int]:
    # out_dir travels in the args rather than being read from the module global:
    # on Windows (spawn) each worker re-imports this module fresh, so a global
    # reassigned in the parent silently does NOT reach the children.
    chunk_path, seed, out_dir = args
    with open(chunk_path, encoding="utf-8") as f:
        rows = [_parse_routed_line(l) for l in f]
    random.Random(seed).shuffle(rows)          # shuffle sources/classes within the chunk

    n = len(rows)
    boards   = np.empty((n, 8, 8, 20), dtype=np.float16)
    policies = np.zeros((n, dc.POLICY_SIZE), dtype=np.float32)
    values   = np.zeros((n, dc.N_WDL),       dtype=np.float32)
    sources  = np.zeros(n,                   dtype=np.uint8)
    n_soft = 0
    for i, (fen, idx, wdl, src, sf) in enumerate(rows):
        b, p, v = dc.encode_sample(fen, idx, wdl, sf_wdl=sf, lam=dc.VALUE_LAMBDA)
        boards[i] = b
        policies[i] = p
        values[i] = v
        sources[i] = src
        n_soft += sf is not None

    cid = os.path.splitext(os.path.basename(chunk_path))[0][1:]  # "c0007" -> "0007"
    out = os.path.join(out_dir, f"chunk_{cid}.npz")
    np.savez_compressed(out, boards=boards, policies=policies, values=values,
                        sources=sources)
    return out, n, n_soft


def assemble(build_dir: str | None = None, out_dir: str | None = None) -> None:
    build_dir = build_dir or BUILD_DIR
    out_dir = out_dir or OUT_DIR
    chunk_files = sorted(glob.glob(os.path.join(build_dir, "c*.txt")))
    work = [(p, SEED + i, out_dir) for i, p in enumerate(chunk_files)]
    tot = soft = 0
    with multiprocessing.Pool(processes=min(ASSEMBLE_WORKERS, len(work))) as pool:
        for out, n, ns in pool.imap_unordered(encode_chunk, work):
            tot += n
            soft += ns
            print(f"  wrote {os.path.basename(out)}  ({n:,} positions, "
                  f"{100*ns/max(n,1):.1f}% Stockfish-blended)", flush=True)
    print(f"\n  {soft:,}/{tot:,} rows ({100*soft/max(tot,1):.1f}%) carry a "
          f"Stockfish-blended value target (lambda={dc.VALUE_LAMBDA}); "
          f"the rest are one-hot (tablebase is exact by design).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble balanced training chunks.")
    ap.add_argument("--route-only", action="store_true",
                    help="run Pass A+B only, leaving _build/ manifests for "
                         "annotate_stockfish.py")
    ap.add_argument("--encode-only", action="store_true",
                    help="run Pass C only, on the manifests already in _build/")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()

    print("=== Dataset Assembler ===")
    print(f"Pools : {POOL_DIR}")
    print(f"Out   : {OUT_DIR}/chunk_*.npz")
    print(f"Value : outcome one-hot blended with Stockfish WDL, "
          f"lambda={dc.VALUE_LAMBDA}\n")

    if not args.encode_only:
        files = _pool_files()
        print("Pass A: counting pools ...", flush=True)
        counts = count_groups(files)
        for s in SOURCES:
            row = {w: counts.get((s, w), 0) for w in (0, 1, 2)}
            tot = sum(row.values())
            print(f"  {s:10s} win {row[0]:>10,}  draw {row[1]:>10,}  "
                  f"loss {row[2]:>10,}  | {tot:>11,}")

        N, q = plan(counts)
        if N <= 0:
            print("\nERROR: pools too small / imbalanced to build even one "
                  "balanced chunk.")
            sys.exit(1)
        print(f"\nPlan: {N} chunks of {CHUNK_SIZE:,}")
        print("  per-chunk quota:")
        for (s, w), per in sorted(q.items()):
            if per > 0:
                print(f"    {s:10s} {['win','draw','loss'][w]:4s}: {per:,}")
        print(f"  totals/chunk: game {SRC_GAME:,} | puzzle {SRC_PUZZLE:,} "
              f"| tb {SRC_TB:,}"
              f"   ||  win {COL_WIN:,} | draw {COL_DRAW:,} | loss {COL_LOSS:,}")

        print(f"\nPass B: routing rows into {N} per-chunk manifests ...", flush=True)
        route(files, counts, N, q)

    if args.route_only:
        print(f"\n=== Routed in {time.time() - t0:.0f}s -> {BUILD_DIR}/ ===")
        print("Next: python supervised_learning/create_dataset/annotate_stockfish.py")
        return

    if not os.path.isdir(BUILD_DIR):
        print(f"ERROR: {BUILD_DIR} not found — run --route-only first.")
        sys.exit(1)

    print(f"Pass C: encoding chunks ({ASSEMBLE_WORKERS} workers) ...", flush=True)
    assemble()

    n_out = len(glob.glob(os.path.join(OUT_DIR, "chunk_*.npz")))
    shutil.rmtree(BUILD_DIR, ignore_errors=True)
    print(f"\n=== Done in {time.time() - t0:.0f}s — {n_out} chunks -> {OUT_DIR}/ ===")


if __name__ == "__main__":
    main()
