"""
PGN → training tensors for BigNetwork supervised learning.

Run from the chess_bot root directory:
    python supervised_learning/process_pgn_data.py

Output: supervised_learning/processed_data/chunk_src_<pgn_stem>_XXXX.npz
Each file contains:
    boards   : float16  (N, 8, 8, 20)
    policies : float32  (N, 1858)   — one-hot move vector
    values   : float32  (N, 3)      — one-hot WDL [win, draw, loss]

Training call:
    net.model.fit(
        data["boards"].astype("float32"),
        {"policy_output": data["policies"], "value_output": data["values"]},
        epochs=1, batch_size=512,
    )
"""

import glob
import multiprocessing
import os
import sys
import time

import chess
import chess.pgn
import numpy as np

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.chdir(_ROOT)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR   = os.path.join(_ROOT, "supervised_learning", "supervised_data")
OUT_DIR    = os.path.join(_ROOT, "supervised_learning", "processed_data")
CHUNK_SIZE = 50_000
N_WORKERS  = 14

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _move_to_index(move: chess.Move, board: chess.Board, lut: dict[str, int]) -> int | None:
    uci = move.uci()
    if board.turn == chess.BLACK:
        uci = uci[0] + str(9 - int(uci[1])) + uci[2] + str(9 - int(uci[3])) + uci[4:]
    if move.promotion == chess.QUEEN:
        uci = uci[:4]
    return lut.get(uci)


def _result_to_wdl(result: str, turn: chess.Color) -> np.ndarray | None:
    wdl = np.zeros(3, dtype=np.float32)
    if result == "1/2-1/2":
        wdl[1] = 1.0
    elif result == "1-0":
        wdl[0 if turn == chess.WHITE else 2] = 1.0
    elif result == "0-1":
        wdl[0 if turn == chess.BLACK else 2] = 1.0
    else:
        return None
    return wdl


def _progress_file(worker_id: int) -> str:
    return os.path.join(OUT_DIR, f"progress_w{worker_id}.txt")


def _load_progress(worker_id: int) -> set[str]:
    path = _progress_file(worker_id)
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def _mark_done(worker_id: int, pgn_basename: str) -> None:
    with open(_progress_file(worker_id), "a") as f:
        f.write(pgn_basename + "\n")

# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def worker_fn(args: tuple) -> tuple[int, int, int, int]:
    """Runs in a subprocess. Returns (worker_id, games, positions, skipped)."""
    worker_id, pgn_files = args

    # Re-initialise in subprocess — Converter can't be pickled safely
    os.chdir(_ROOT)
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    from reinforcement_learning.helpers.converter import Converter  # noqa: E402
    converter = Converter()
    lut = {v: int(k) for k, v in converter.lookup.items()}

    done = _load_progress(worker_id)
    remaining = [p for p in pgn_files if os.path.basename(p) not in done]
    if not remaining:
        return worker_id, 0, 0, 0

    boards_buf   = np.empty((CHUNK_SIZE, 8, 8, 20), dtype=np.float16)
    policies_buf = np.zeros((CHUNK_SIZE, 1858),       dtype=np.float32)
    values_buf   = np.empty((CHUNK_SIZE, 3),           dtype=np.float32)
    buf_idx = 0

    # Chunks are named after the source PGN and numbered from zero within each
    # file (both set at the top of each file's loop below). This makes a file's
    # chunks identifiable by name, so an interrupted file's partial chunks can
    # be deleted before it is reprocessed — even if round-robin re-assigns the
    # file to a different worker on resume.
    cur_stem = ""
    file_chunk_idx = 0

    def _save():
        nonlocal buf_idx, file_chunk_idx
        name = f"chunk_src_{cur_stem}_{file_chunk_idx:04d}.npz"
        np.savez_compressed(
            os.path.join(OUT_DIR, name),
            boards=boards_buf[:buf_idx],
            policies=policies_buf[:buf_idx],
            values=values_buf[:buf_idx],
        )
        print(f"  [w{worker_id}] {name}  {buf_idx:,} positions", flush=True)
        file_chunk_idx += 1
        buf_idx = 0

    total_games = total_pos = skipped = 0

    for pgn_path in remaining:
        pgn_name = os.path.basename(pgn_path)
        cur_stem = os.path.splitext(pgn_name)[0]
        file_chunk_idx = 0

        # Drop any chunks left from an interrupted run of this file. A file is
        # only marked done once fully flushed, so chunks on disk for a not-done
        # file are stale (possibly written by another worker before round-robin
        # re-assigned the file here). Removing them first prevents duplication.
        for stale in glob.glob(os.path.join(OUT_DIR, f"chunk_src_{glob.escape(cur_stem)}_*.npz")):
            os.remove(stale)

        print(f"  [w{worker_id}] {pgn_name} ...", flush=True)
        with open(pgn_path, encoding="utf-8", errors="replace") as fh:
            while True:
                try:
                    game = chess.pgn.read_game(fh)
                except Exception:
                    continue
                if game is None:
                    break

                result = game.headers.get("Result", "*")
                total_games += 1
                board = game.board()

                for move in game.mainline_moves():
                    wdl = _result_to_wdl(result, board.turn)
                    move_idx = _move_to_index(move, board, lut)

                    if wdl is not None and move_idx is not None:
                        boards_buf[buf_idx]          = converter.board_to_input_tensor(board)
                        policies_buf[buf_idx]        = 0.0
                        policies_buf[buf_idx, move_idx] = 1.0
                        values_buf[buf_idx]          = wdl
                        buf_idx += 1
                        total_pos += 1

                        if buf_idx >= CHUNK_SIZE:
                            _save()
                    else:
                        skipped += 1

                    board.push(move)

        # Flush this file's buffered positions before marking it done, so the
        # resume unit (a whole PGN file) matches the data unit (a chunk).
        # Otherwise an interrupted file is reprocessed — and round-robin may
        # hand it to a different worker — on resume, duplicating positions that
        # were already saved to disk.
        if buf_idx:
            _save()
        _mark_done(worker_id, pgn_name)

    return worker_id, total_games, total_pos, skipped

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== PGN Data Processor ===")
    print(f"Data    : {DATA_DIR}")
    print(f"Out     : {OUT_DIR}")
    print(f"Workers : {N_WORKERS}\n")

    os.makedirs(OUT_DIR, exist_ok=True)

    pgn_files = sorted(glob.glob(os.path.join(DATA_DIR, "Batch*.pgn")))
    if not pgn_files:
        print(f"ERROR: no Batch*.pgn files found in {DATA_DIR}")
        sys.exit(1)

    # Check overall resume state
    all_done: set[str] = set()
    for wid in range(N_WORKERS):
        all_done |= _load_progress(wid)
    remaining_total = [p for p in pgn_files if os.path.basename(p) not in all_done]

    print(f"Found {len(pgn_files)} PGN files — {len(all_done)} already done, {len(remaining_total)} remaining\n")

    if not remaining_total:
        total_chunks = len(glob.glob(os.path.join(OUT_DIR, "chunk_src_*.npz")))
        print(f"All done — {total_chunks} chunk file(s) in {OUT_DIR}/")
        sys.exit(0)

    # Distribute remaining files round-robin across workers
    worker_files: list[list[str]] = [[] for _ in range(N_WORKERS)]
    for i, path in enumerate(remaining_total):
        worker_files[i % N_WORKERS].append(path)

    work_items = [(wid, files) for wid, files in enumerate(worker_files) if files]

    t0 = time.time()
    with multiprocessing.Pool(processes=N_WORKERS) as pool:
        results = pool.map(worker_fn, work_items)

    total_games = sum(r[1] for r in results)
    total_pos   = sum(r[2] for r in results)
    skipped     = sum(r[3] for r in results)
    total_chunks = len(glob.glob(os.path.join(OUT_DIR, "chunk_src_*.npz")))
    elapsed = time.time() - t0

    print(f"\n=== Done in {elapsed:.0f}s ===")
    print(f"  {total_games:,} games | {total_pos:,} positions | {skipped} skipped")
    print(f"  {total_chunks} chunk file(s) in {OUT_DIR}/")
