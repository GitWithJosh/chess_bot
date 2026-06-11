"""
PGN → training tensors for BigNetwork supervised learning.

Run from the chess_bot root directory:
    python supervised_learning/process_pgn_data.py

Output: supervised_learning/processed_data/chunk_XXXX.npz
Each file contains:
    boards   : float16  (N, 8, 8, 112)
    policies : float32  (N, 1858)   — soft move distribution per unique position
    values   : float32  (N, 3)      — soft WDL [win, draw, loss] per unique position

Training call (bypass BigNetwork.train() since values are already soft WDL):
    net.model.fit(
        data["boards"].astype("float32"),
        {"policy_output": data["policies"], "value_output": data["values"]},
        epochs=1, batch_size=512,
    )

Memory usage: ~200-400 MB RAM (aggregation lives in a temp SQLite DB on disk).
"""

import glob
import os
import sqlite3
import sys
import time

import chess
import chess.pgn
import numpy as np

# --- Path setup: must be run from chess_bot root; sys.path lets IDE resolve the import ---
_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Converter opens "reinforcement_learning/move_lookup.json" relative to CWD.
os.chdir(_ROOT)

from reinforcement_learning.helpers.converter import Converter  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(_ROOT, "supervised_learning", "supervised_data")
OUT_DIR  = os.path.join(_ROOT, "supervised_learning", "processed_data")
TEMP_DB  = os.path.join(OUT_DIR, "temp_aggregate.db")

CHUNK_SIZE  = 50_000   # positions per output .npz (tune down if RAM is tight)
BATCH_SIZE  = 5_000    # SQLite UPSERT batch size per transaction

# ---------------------------------------------------------------------------
# Move helpers (inlined to avoid depending on Converter internals)
# ---------------------------------------------------------------------------

def _mirror_uci(uci: str) -> str:
    """Flip ranks so a Black move maps to White's-perspective lookup key."""
    flipped = uci[0] + str(9 - int(uci[1])) + uci[2] + str(9 - int(uci[3]))
    return flipped + uci[4:] if len(uci) > 4 else flipped


def _move_to_lut_key(move: chess.Move, board: chess.Board) -> str:
    """Return the string key used in move_lookup.json for this move."""
    uci = move.uci()
    if board.turn == chess.BLACK:
        uci = _mirror_uci(uci)
    if move.promotion == chess.QUEEN:
        uci = uci[:4]   # queen promotions stored without 'q' in the lookup
    return uci


def _outcome(result: str, turn: chess.Color):
    """Return (wins, draws, losses) contribution of this game for the current player."""
    if result == "1/2-1/2":
        return 0, 1, 0
    if result == "1-0":
        return (1, 0, 0) if turn == chess.WHITE else (0, 0, 1)
    if result == "0-1":
        return (1, 0, 0) if turn == chess.BLACK else (0, 0, 1)
    return None   # game not finished / unknown

# ---------------------------------------------------------------------------
# Pass 1 — stream PGNs, aggregate (fen, move_idx) → WDL counts in SQLite
# ---------------------------------------------------------------------------

def pass1_aggregate(pgn_files: list[str], lut: dict[str, int]) -> None:
    """
    For every position in every game write one UPSERT to SQLite.
    After this pass the DB holds one row per unique (position, move) pair
    with cumulative win/draw/loss counts.

    RAM usage: SQLite cache (~64 MB) + UPSERT batch (~a few MB) = well under 200 MB.
    Disk usage: ~300–600 MB for the temp DB.
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    conn = sqlite3.connect(TEMP_DB)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-65536")   # 64 MB page cache

    conn.execute("""
        CREATE TABLE IF NOT EXISTS agg (
            fen      TEXT    NOT NULL,
            move_idx INTEGER NOT NULL,
            wins     INTEGER NOT NULL DEFAULT 0,
            draws    INTEGER NOT NULL DEFAULT 0,
            losses   INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (fen, move_idx)
        )
    """)
    conn.commit()

    upsert = """
        INSERT INTO agg (fen, move_idx, wins, draws, losses)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(fen, move_idx) DO UPDATE SET
            wins   = wins   + excluded.wins,
            draws  = draws  + excluded.draws,
            losses = losses + excluded.losses
    """

    batch: list[tuple] = []
    total_games = 0
    total_pos   = 0
    skipped     = 0
    t0 = time.time()

    def _flush() -> None:
        conn.executemany(upsert, batch)
        conn.commit()
        batch.clear()

    for pgn_path in pgn_files:
        print(f"  {os.path.basename(pgn_path)} ...", flush=True)
        with open(pgn_path, encoding="utf-8", errors="replace") as fh:
            while True:
                try:
                    game = chess.pgn.read_game(fh)
                except Exception:
                    continue
                if game is None:
                    break

                result = game.headers.get("Result", "*")
                contrib = None   # evaluated lazily per-move
                total_games += 1
                board = game.board()

                for move in game.mainline_moves():
                    contrib = _outcome(result, board.turn)
                    if contrib is not None:
                        lut_key = _move_to_lut_key(move, board)
                        idx = lut.get(lut_key)
                        if idx is not None:
                            fen = " ".join(board.fen().split()[:5])   # omit fullmove number
                            batch.append((fen, idx, *contrib))
                        else:
                            skipped += 1
                    total_pos += 1
                    board.push(move)

                if len(batch) >= BATCH_SIZE:
                    _flush()

    if batch:
        _flush()
    conn.close()

    elapsed = time.time() - t0
    (n_unique,) = sqlite3.connect(TEMP_DB).execute(
        "SELECT COUNT(DISTINCT fen) FROM agg"
    ).fetchone()

    print(f"\n  {total_games:,} games | {total_pos:,} positions | {skipped} skipped  "
          f"({elapsed:.0f}s)")
    print(f"  Unique positions: {n_unique:,}")


# ---------------------------------------------------------------------------
# Pass 2 — read aggregated DB, build tensors, save .npz chunks
# ---------------------------------------------------------------------------

def pass2_tensors(converter: Converter) -> int:
    """
    Iterates the aggregated SQLite table ordered by FEN (uses the primary-key
    B-tree — no in-memory sort needed) and groups rows by position.

    For each unique position:
        • reconstructs chess.Board from the stored FEN (no move-stack → history planes = 0)
        • builds soft policy vector from move-count frequencies
        • builds soft WDL from win/draw/loss fractions

    Saves chunks of CHUNK_SIZE positions as compressed .npz files.
    Peak RAM: one pre-allocated chunk ≈ CHUNK_SIZE × (14 KB + 7.5 KB) ≈ 1 GB for 50k.
    """
    conn = sqlite3.connect(TEMP_DB)
    cursor = conn.execute(
        "SELECT fen, move_idx, wins, draws, losses FROM agg ORDER BY fen"
    )

    # Pre-allocate chunk buffers (avoids list-of-arrays overhead)
    boards_buf   = np.empty((CHUNK_SIZE, 8, 8, 112), dtype=np.float16)
    policies_buf = np.empty((CHUNK_SIZE, 1858),       dtype=np.float32)
    values_buf   = np.empty((CHUNK_SIZE, 3),           dtype=np.float32)
    buf_idx      = 0
    chunk_idx    = 0

    def _save() -> None:
        nonlocal buf_idx, chunk_idx
        path = os.path.join(OUT_DIR, f"chunk_{chunk_idx:04d}.npz")
        np.savez_compressed(
            path,
            boards=boards_buf[:buf_idx],
            policies=policies_buf[:buf_idx],
            values=values_buf[:buf_idx],
        )
        print(f"  chunk_{chunk_idx:04d}.npz  {buf_idx:,} positions", flush=True)
        chunk_idx += 1
        buf_idx = 0

    # Group rows by FEN — the ORDER BY fen on a PK-indexed table is free
    cur_fen = None
    cur_moves: dict[int, int] = {}   # {move_idx: total_appearances}
    cur_w = cur_d = cur_l = 0

    def _emit() -> None:
        nonlocal buf_idx, cur_fen, cur_w, cur_d, cur_l
        try:
            board = chess.Board(cur_fen)
        except ValueError:
            return

        # Board tensor (history planes are zero — no move stack)
        boards_buf[buf_idx] = converter.board_to_input_tensor(board)

        # Soft policy
        total_m = sum(cur_moves.values())
        policies_buf[buf_idx] = 0.0
        for idx, cnt in cur_moves.items():
            policies_buf[buf_idx, idx] = cnt / total_m

        # Soft WDL
        total_g = cur_w + cur_d + cur_l
        if total_g:
            values_buf[buf_idx] = [cur_w / total_g, cur_d / total_g, cur_l / total_g]
        else:
            values_buf[buf_idx] = [1 / 3, 1 / 3, 1 / 3]

        buf_idx += 1

    for fen, move_idx, wins, draws, losses in cursor:
        if fen != cur_fen:
            if cur_fen is not None:
                _emit()
                cur_moves.clear()
                cur_w = cur_d = cur_l = 0
                if buf_idx >= CHUNK_SIZE:
                    _save()
            cur_fen = fen

        cur_moves[move_idx] = wins + draws + losses   # unique per (fen, move_idx)
        cur_w += wins
        cur_d += draws
        cur_l += losses

    if cur_fen is not None:
        _emit()
    if buf_idx:
        _save()

    conn.close()
    return chunk_idx


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== PGN Data Processor ===")
    print(f"Data : {DATA_DIR}")
    print(f"Out  : {OUT_DIR}\n")

    converter = Converter()
    lut = {v: int(k) for k, v in converter.lookup.items()}

    pgn_files = sorted(glob.glob(os.path.join(DATA_DIR, "Batch*.pgn")))
    if not pgn_files:
        print(f"ERROR: no Batch*.pgn files found in {DATA_DIR}")
        sys.exit(1)
    print(f"Found {len(pgn_files)} PGN files\n")

    print("Pass 1 — aggregating positions into SQLite ...")
    pass1_aggregate(pgn_files, lut)

    print("\nPass 2 — building tensors ...")
    n_chunks = pass2_tensors(converter)

    os.remove(TEMP_DB)
    print(f"\nDone — {n_chunks} chunk file(s) in {OUT_DIR}/")
