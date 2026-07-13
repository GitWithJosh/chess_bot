"""
Lichess puzzle CSV -> puzzle manifest (fen,move_idx,wdl rows).

Run from the chess_bot root:
    python supervised_learning/create_dataset/process_puzzle_data.py          # full run
    python supervised_learning/create_dataset/process_puzzle_data.py 50000    # test: first 50k CSV rows

Puzzle convention (Lichess): the FEN is the position BEFORE the opponent's
setup move. Moves[0] is that setup move (the opponent's blunder reply); the
SOLVER is to move after it, and plays Moves[1], Moves[3], ...  We record:

  - SOLVER-to-move positions (before Moves[1,3,...]) labeled WIN — the solver
    is, by construction, in a winning/crushing position;
  - DEFENDER-to-move positions (before Moves[2,4,...]) labeled LOSS, with the
    line's engine-chosen defensive reply as the policy target. For mate
    puzzles the defender is provably lost; for endgame-technique puzzles
    "lost" is the puzzle generator's engine judgement, not a proof (accepted
    noise). Defender rows give the value head BOTH sides of a tactic, so
    "looks like a puzzle" no longer implies "side to move wins" — the probe-B
    shortcut (see inspect/probe_value_head.py).

Selection: mate*/endgame-themed puzzles first (they directly teach finishing);
defender rows are taken from priority puzzles only. If the win capacity is not
filled by priority puzzles, a second pass fills it with other themes (wins
only). Puzzles listed in results/probes/positions/probe_puzzle_ids.txt are
skipped ENTIRELY so probe B stays held-out now that defender positions are
trained on.

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

# Puzzle ids used by the value-head probes — excluded from training so the
# probe stays a clean held-out set. Written by inspect/probe_value_head.py.
PROBE_IDS_PATH = os.path.join(dc.repo_root(), "supervised_learning", "results",
                              "probes", "positions", "probe_puzzle_ids.txt")

# Per-class caps. The assembler needs 2,500 win + 2,500 loss per 50k chunk, so
# 1.5M each covers ~600 chunks with margin for its random sampling.
MAX_WIN  = 1_500_000
MAX_LOSS = 1_500_000

PRIORITY_THEMES = ("mate", "endgame")  # substrings; "mate" also catches mateInN


def _is_priority(themes: str) -> bool:
    return any(t in themes for t in PRIORITY_THEMES)


def _probe_ids() -> set[str]:
    if os.path.exists(PROBE_IDS_PATH):
        with open(PROBE_IDS_PATH, encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}
    return set()


def _puzzle_positions(fen: str, moves: list[str], lut: dict[str, int],
                      want_defenders: bool):
    """Yield (fen, move_idx, wdl) for each ply of the puzzle line.

    Solver plies (odd i) -> WIN rows; defender plies (even i >= 2) -> LOSS rows
    (only when want_defenders). Stops at the first malformed/illegal move.
    """
    board = chess.Board(fen)
    try:
        board.push_uci(moves[0])          # opponent setup move -> solver to move
    except Exception:
        return
    for i in range(1, len(moves)):
        try:
            move = chess.Move.from_uci(moves[i])
        except Exception:
            return
        if move not in board.legal_moves:
            return                        # malformed line; stop this puzzle
        is_solver = (i % 2 == 1)          # 1,3,5,... are the solver's moves
        if is_solver or want_defenders:
            idx = dc.move_to_index(move, board, lut)
            if idx is not None:
                yield board.fen(), idx, (dc.WDL_WIN if is_solver else dc.WDL_LOSS)
        board.push(move)


def _stream(row_limit: int | None, want_priority: bool, lut, out,
            written: dict, skip_ids: set[str]) -> None:
    """One pass over the CSV writing positions of the requested priority class.

    Priority pass: solver (win) + defender (loss) rows. Fill pass: wins only.
    """
    csv.field_size_limit(10 ** 7)
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for n, row in enumerate(reader):
            if row_limit is not None and n >= row_limit:
                break
            if _is_priority(row["Themes"]) != want_priority:
                continue
            if row["PuzzleId"] in skip_ids:
                written["probe_skipped"] += 1
                continue
            moves = row["Moves"].split()
            if len(moves) < 2:
                continue
            for fen, idx, wdl in _puzzle_positions(
                    row["FEN"], moves, lut, want_defenders=want_priority):
                if wdl == dc.WDL_WIN and written["win"] < MAX_WIN:
                    out.write(dc.manifest_line(fen, idx, wdl))
                    written["win"] += 1
                elif wdl == dc.WDL_LOSS and written["loss"] < MAX_LOSS:
                    out.write(dc.manifest_line(fen, idx, wdl))
                    written["loss"] += 1
            win_full = written["win"] >= MAX_WIN
            loss_full = (not want_priority) or written["loss"] >= MAX_LOSS
            if win_full and loss_full:
                return
            if n % 500_000 == 0 and n:
                print(f"  ...scanned {n:,} rows, written "
                      f"{written['win']:,} win / {written['loss']:,} loss", flush=True)


def main() -> None:
    row_limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    os.makedirs(OUT_DIR, exist_ok=True)
    lut = dc.load_move_lut()
    skip_ids = _probe_ids()

    print("=== Puzzle Data Processor ===")
    print(f"CSV : {CSV_PATH}")
    print(f"Out : {OUT_PATH}")
    print(f"Caps: {MAX_WIN:,} win (solver) / {MAX_LOSS:,} loss (defender)"
          + (f"  | TEST: first {row_limit:,} CSV rows" if row_limit else ""))
    print(f"Probe exclusion: {len(skip_ids):,} puzzle ids"
          + ("" if skip_ids else "  (probe_puzzle_ids.txt not found)"))

    t0 = time.time()
    written = {"win": 0, "loss": 0, "probe_skipped": 0}
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        _stream(row_limit, want_priority=True, lut=lut, out=out,
                written=written, skip_ids=skip_ids)
        print(f"  priority (mate/endgame): {written['win']:,} win / "
              f"{written['loss']:,} loss positions", flush=True)
        if written["win"] < MAX_WIN:
            _stream(row_limit, want_priority=False, lut=lut, out=out,
                    written=written, skip_ids=skip_ids)
            print(f"  after fill (others, wins only): {written['win']:,} win", flush=True)

    print(f"=== Done in {time.time() - t0:.0f}s — {written['win']:,} win + "
          f"{written['loss']:,} loss -> {OUT_PATH} "
          f"({written['probe_skipped']:,} probe puzzles skipped) ===")


if __name__ == "__main__":
    main()
