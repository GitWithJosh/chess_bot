"""
Comprehensive chunk quality analysis for supervised learning data.

Metrics per chunk:
  Board uniqueness:
    unique_boards      — exact count of distinct board states
    unique_board_move  — exact count of distinct (board, move) pairs
    dup_ratio          — fraction of positions that are redundant board states
    natural_dup_ratio  — same board, different move (expected: popular openings)
    game_rep_ratio     — same board AND same move (bad: repeated game segments)

  Policy diversity:
    unique_moves       — number of distinct moves played across the chunk
    policy_entropy     — entropy (bits) over the move frequency distribution
    top5_moves         — most common move indices with counts and frequencies

  Value balance:
    win/draw/loss_pct  — WDL distribution
    wdl_entropy        — entropy (bits) over WDL; low = skewed result labels

  Validity:
    invalid_moves      — positions with move index outside [0, 1857]
    n                  — chunk size (< CHUNK_SIZE means tail/partial chunk)

Red flags emitted:
  GAME_REP      game_rep_ratio > 10%   (suspected repeated game segments)
  HIGH_DUP      total dup_ratio > 50%
  DOMINANT_MOVE top-1 move freq > 15%
  SKEWED_WDL    wdl_entropy < 1.0 bit
  FEW_MOVES     unique_moves < 200
  INVALID_MOVES any move index out of range
  PARTIAL       n < CHUNK_SIZE

Run from chess_bot root:
    python supervised_learning/analyse_chunks.py
"""

import gc
import glob
import json
import multiprocessing
import os
import time
import traceback as tb

import numpy as np


DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_data")
OUT_FILE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chunk_analysis.json")
N_WORKERS  = 6
CHUNK_SIZE = 50_000


# ---------------------------------------------------------------------------
# Per-chunk analysis
# ---------------------------------------------------------------------------

def _row_hashes(arr2d: np.ndarray) -> np.ndarray:
    """Exact per-row hash using Python's built-in hash() on raw bytes.

    Returns int64 array of length N.  Hash seed is fixed per process (Python
    PYTHONHASHSEED), so hashes are consistent within one run but differ across
    runs — fine for within-chunk uniqueness counting.
    """
    n   = len(arr2d)
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        out[i] = hash(arr2d[i].tobytes())
    return out


def analyse_chunk(path: str) -> dict:
    fname = os.path.basename(path)
    try:
        with np.load(path) as npz:

            # ------------------------------------------------------------------
            # Boards  (float16, kept as-is to halve RAM vs float32)
            # ------------------------------------------------------------------
            boards = npz["boards"]           # (N, 8, 8, 20) float16
            n      = len(boards)
            flat   = boards.reshape(n, -1)   # (N, 1280) — view, no copy
            board_hashes = _row_hashes(flat)
            del boards, flat
            gc.collect()

            # ------------------------------------------------------------------
            # Policies
            # ------------------------------------------------------------------
            policies     = npz["policies"]                        # (N, 1858) float32
            move_indices = np.argmax(policies, axis=1).astype(np.int32)  # (N,)
            invalid_moves = int(np.sum((move_indices < 0) | (move_indices >= 1858)))

            unique_moves, move_counts = np.unique(move_indices, return_counts=True)
            n_unique_moves = int(len(unique_moves))
            top_order      = np.argsort(-move_counts)
            top5_moves     = [
                {
                    "move_idx": int(unique_moves[i]),
                    "count":    int(move_counts[i]),
                    "freq":     round(float(move_counts[i]) / n, 4),
                }
                for i in top_order[:5]
            ]
            move_probs     = move_counts / move_counts.sum()
            policy_entropy = float(-np.sum(move_probs * np.log2(move_probs + 1e-12)))
            del policies, move_counts, move_probs
            gc.collect()

            # ------------------------------------------------------------------
            # Values  (WDL one-hot: [win, draw, loss])
            # ------------------------------------------------------------------
            values        = npz["values"]             # (N, 3) float32
            value_classes = np.argmax(values, axis=1) # 0=win 1=draw 2=loss
            win_count     = int(np.sum(value_classes == 0))
            draw_count    = int(np.sum(value_classes == 1))
            loss_count    = int(np.sum(value_classes == 2))
            wdl_probs     = np.array([win_count, draw_count, loss_count],
                                     dtype=np.float64) / n
            wdl_entropy   = float(-np.sum(wdl_probs * np.log2(wdl_probs + 1e-12)))
            del values, value_classes, wdl_probs
            gc.collect()

        # ----------------------------------------------------------------------
        # Uniqueness decomposition
        # ----------------------------------------------------------------------
        n_unique_boards     = len(set(board_hashes.tolist()))
        board_move_set      = set(zip(board_hashes.tolist(), move_indices.tolist()))
        n_unique_board_move = len(board_move_set)
        del board_hashes, board_move_set

        # total_dup      = all redundant board states
        # natural_dup    = same board, different move  (popular openings — expected)
        # game_rep_dup   = same board AND same move    (repeated game segments — bad)
        total_dup    = n - n_unique_boards
        natural_dup  = n_unique_board_move - n_unique_boards
        game_rep_dup = n - n_unique_board_move

        dup_ratio         = round(total_dup    / n, 4)
        natural_dup_ratio = round(natural_dup  / n, 4)
        game_rep_ratio    = round(game_rep_dup / n, 4)

        # ----------------------------------------------------------------------
        # Flags
        # ----------------------------------------------------------------------
        flags = []
        if game_rep_ratio > 0.10:
            flags.append(f"GAME_REP:{game_rep_ratio:.1%}")
        if dup_ratio > 0.50:
            flags.append(f"HIGH_DUP:{dup_ratio:.1%}")
        top1_freq = top5_moves[0]["freq"] if top5_moves else 0.0
        if top1_freq > 0.15:
            flags.append(f"DOMINANT_MOVE:{top1_freq:.1%}")
        if wdl_entropy < 1.0:
            flags.append(f"SKEWED_WDL:W{win_count/n:.0%}D{draw_count/n:.0%}L{loss_count/n:.0%}")
        if n_unique_moves < 200:
            flags.append(f"FEW_MOVES:{n_unique_moves}")
        if invalid_moves > 0:
            flags.append(f"INVALID_MOVES:{invalid_moves}")
        if n < CHUNK_SIZE:
            flags.append(f"PARTIAL:{n}")

        return {
            "file":               fname,
            "n":                  n,
            "unique_boards":      n_unique_boards,
            "unique_board_move":  n_unique_board_move,
            "dup_ratio":          dup_ratio,
            "natural_dup_ratio":  natural_dup_ratio,
            "game_rep_ratio":     game_rep_ratio,
            "unique_moves":       n_unique_moves,
            "policy_entropy":     round(policy_entropy, 3),
            "top5_moves":         top5_moves,
            "win_pct":            round(win_count  / n, 4),
            "draw_pct":           round(draw_count / n, 4),
            "loss_pct":           round(loss_count / n, 4),
            "wdl_entropy":        round(wdl_entropy, 3),
            "invalid_moves":      invalid_moves,
            "flags":              flags,
            "error":              None,
        }

    except Exception as e:
        return {
            "file":  fname,
            "error": str(e),
            "trace": tb.format_exc(),
            "flags": ["ERROR"],
        }


# ---------------------------------------------------------------------------
# Multiprocessing glue
# ---------------------------------------------------------------------------

def _worker(args):
    idx, path = args
    return idx, analyse_chunk(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    chunks = sorted(
        glob.glob(os.path.join(DATA_DIR, "**", "chunk_*.npz"), recursive=True)
    )
    chunks = [c for c in chunks if os.path.basename(c).startswith("chunk_")]
    if not chunks:
        print(f"No chunk_*.npz found in {DATA_DIR}")
        return

    print(f"Analysing {len(chunks)} chunks with {N_WORKERS} workers")
    print(f"Peak RAM per worker ≈ 700 MB (boards float16).  "
          f"Total ≈ {N_WORKERS * 0.7:.0f} GB — reduce N_WORKERS if you see OOM.\n")

    results = [None] * len(chunks)
    t0 = time.time()
    done = 0

    with multiprocessing.Pool(processes=N_WORKERS) as pool:
        for idx, result in pool.imap_unordered(_worker, enumerate(chunks), chunksize=2):
            results[idx] = result
            done += 1
            if done % 20 == 0 or done == len(chunks):
                elapsed = time.time() - t0
                eta     = elapsed / done * (len(chunks) - done)
                line    = f"  {done}/{len(chunks)}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)"
                if result.get("flags"):
                    line += f"  ← {result['file']} {result['flags']}"
                print(line, flush=True)

    results = [r for r in results if r is not None]
    results.sort(key=lambda r: r["file"])

    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    # --------------------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------------------
    errors     = [r for r in results if r.get("error")]
    game_rep   = [r for r in results if r.get("game_rep_ratio",  0) > 0.10]
    high_dup   = [r for r in results if r.get("dup_ratio",       0) > 0.50]
    partial    = [r for r in results if r.get("n", CHUNK_SIZE)       < CHUNK_SIZE]
    skewed_wdl = [r for r in results if r.get("wdl_entropy",   99) < 1.0]
    few_moves  = [r for r in results if r.get("unique_moves",  999) < 200]
    invalid    = [r for r in results if r.get("invalid_moves",   0)  > 0]

    elapsed = time.time() - t0
    print(f"\n=== Done in {elapsed:.0f}s  ({elapsed/len(chunks):.1f}s/chunk avg) ===")
    print(f"Total chunks          : {len(results)}")
    print(f"Errors                : {len(errors)}")
    print(f"Partial chunks        : {len(partial)}")
    print(f"Game-rep dups > 10%   : {len(game_rep)}  ← suspected repeated-game segments")
    print(f"High total dup > 50%  : {len(high_dup)}")
    print(f"Skewed WDL            : {len(skewed_wdl)}")
    print(f"Few unique moves       : {len(few_moves)}")
    print(f"Invalid move indices   : {len(invalid)}")

    if game_rep:
        print(f"\n--- Top game-repetition offenders ---")
        for r in sorted(game_rep, key=lambda x: -x.get("game_rep_ratio", 0))[:30]:
            print(
                f"  {r['file']:20s}  game_rep={r['game_rep_ratio']:.1%}"
                f"  total_dup={r['dup_ratio']:.1%}"
                f"  unique_boards={r.get('unique_boards', '?'):,}"
            )

    if errors:
        print(f"\n--- Errors ---")
        for r in errors:
            print(f"  {r['file']}: {r['error']}")

    print(f"\nFull results → {OUT_FILE}")


if __name__ == "__main__":
    main()
