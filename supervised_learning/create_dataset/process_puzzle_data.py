"""
Lichess puzzle CSV -> puzzle manifest (fen,move_idx,wdl rows).

Run from the chess_bot root:
    python supervised_learning/create_dataset/process_puzzle_data.py          # full run
    python supervised_learning/create_dataset/process_puzzle_data.py 50000    # test: first 50k CSV rows

Puzzle convention (Lichess): the FEN is the position BEFORE the opponent's
setup move. Moves[0] is that setup move (the opponent's blunder reply); the
SOLVER is to move after it, and plays Moves[1], Moves[3], ...  We therefore:
  - push Moves[0],
  - record (position, solver move) for every solver-to-move ply,
  - label each as a WIN for the side to move (the solver is, by construction,
    in a winning/crushing tactic). Opponent-reply positions are NOT recorded.

Selection: mate*/endgame-themed puzzles first (they directly teach finishing),
then fill with others, up to MAX_POSITIONS. mate+endgame puzzles already vastly
exceed the cap, so in practice the fill pass rarely runs.

Output: supervised_learning/pools/pool_puzzles.csv
"""

import csv
import os
import sys
import time

import chess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_common as dc  # noqa: E402

CSV_PATH   = os.path.join(dc.repo_root(), "supervised_learning", "supervised_data", "lichess_db_puzzle.csv")
OUT_DIR    = os.path.join(dc.repo_root(), "supervised_learning", "pools")
OUT_PATH   = os.path.join(OUT_DIR, "pool_puzzles.csv")

# Cap on solver positions written. ~10% of a 50k chunk = 5k/chunk, so 3M covers
# ~600 chunks of puzzle slots with margin for the assembler's random sampling.
MAX_POSITIONS = 3_000_000

PRIORITY_THEMES = ("mate", "endgame")  # substrings; "mate" also catches mateInN


def _is_priority(themes: str) -> bool:
    return any(t in themes for t in PRIORITY_THEMES)


def _solver_positions(fen: str, moves: list[str], lut: dict[str, int]):
    """Yield (fen, move_idx, WDL_WIN) for each solver-to-move ply, or nothing on error."""
    board = chess.Board(fen)
    try:
        board.push_uci(moves[0])          # opponent setup move -> solver is now to move
    except Exception:
        return
    for i in range(1, len(moves)):
        try:
            move = chess.Move.from_uci(moves[i])
        except Exception:
            return
        is_solver_ply = (i % 2 == 1)      # 1,3,5,... are the solver's moves
        if is_solver_ply:
            idx = dc.move_to_index(move, board, lut)
            if idx is not None:
                yield board.fen(), idx, dc.WDL_WIN
        if move not in board.legal_moves:
            return                        # malformed line; stop this puzzle
        board.push(move)


def _stream(row_limit: int | None, want_priority: bool, lut, out, written: int) -> int:
    """One pass over the CSV writing positions of the requested priority class."""
    csv.field_size_limit(10 ** 7)
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for n, row in enumerate(reader):
            if row_limit is not None and n >= row_limit:
                break
            if _is_priority(row["Themes"]) != want_priority:
                continue
            moves = row["Moves"].split()
            if len(moves) < 2:
                continue
            for fen, idx, wdl in _solver_positions(row["FEN"], moves, lut):
                out.write(dc.manifest_line(fen, idx, wdl))
                written += 1
                if written >= MAX_POSITIONS:
                    return written
            if n % 500_000 == 0 and n:
                print(f"  ...scanned {n:,} rows, written {written:,}", flush=True)
    return written


def main() -> None:
    row_limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    os.makedirs(OUT_DIR, exist_ok=True)
    lut = dc.load_move_lut()

    print("=== Puzzle Data Processor ===")
    print(f"CSV : {CSV_PATH}")
    print(f"Out : {OUT_PATH}")
    print(f"Cap : {MAX_POSITIONS:,} solver positions"
          + (f"  | TEST: first {row_limit:,} CSV rows" if row_limit else ""))

    t0 = time.time()
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        written = _stream(row_limit, want_priority=True, lut=lut, out=out, written=0)
        print(f"  priority (mate/endgame): {written:,} positions", flush=True)
        if written < MAX_POSITIONS:
            written = _stream(row_limit, want_priority=False, lut=lut, out=out, written=written)
            print(f"  after fill (others)    : {written:,} positions", flush=True)

    print(f"=== Done in {time.time() - t0:.0f}s — {written:,} positions -> {OUT_PATH} ===")


if __name__ == "__main__":
    main()
