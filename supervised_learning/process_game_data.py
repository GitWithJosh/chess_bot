"""
PGN games -> game manifest shards (fen,move_idx,wdl rows).

Run from the chess_bot root:
    python supervised_learning/process_game_data.py            # full run
    python supervised_learning/process_game_data.py 3          # test: first 3 PGN files

Same parsing/labelling as process_pgn_data.py (every mainline position, WDL from
the game result relative to the side to move), but instead of writing encoded
chunks it emits the lightweight manifest the assembler (build_dataset.py) mixes
and balances. Board encoding is deferred to the assembler so it only runs on the
positions actually kept after 33/33/33 balancing — not all ~22M.

Output: supervised_learning/pools/pool_games_w{worker}.csv
"""

import glob
import multiprocessing
import os
import sys
import time

import chess
import chess.pgn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_common as dc  # noqa: E402

DATA_DIR  = os.path.join(dc.repo_root(), "supervised_learning", "supervised_data")
OUT_DIR   = os.path.join(dc.repo_root(), "supervised_learning", "pools")
N_WORKERS = 14


def worker_fn(args) -> tuple[int, int, int, int]:
    worker_id, pgn_files = args
    lut = dc.load_move_lut()
    out_path = os.path.join(OUT_DIR, f"pool_games_w{worker_id}.csv")

    games = positions = skipped = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for pgn_path in pgn_files:
            with open(pgn_path, encoding="utf-8", errors="replace") as fh:
                while True:
                    try:
                        game = chess.pgn.read_game(fh)
                    except Exception:
                        continue
                    if game is None:
                        break
                    result = game.headers.get("Result", "*")
                    games += 1
                    board = game.board()
                    for move in game.mainline_moves():
                        wdl = dc.result_to_wdl_class(result, board.turn)
                        idx = dc.move_to_index(move, board, lut)
                        if wdl is not None and idx is not None:
                            out.write(dc.manifest_line(board.fen(), idx, wdl))
                            positions += 1
                        else:
                            skipped += 1
                        board.push(move)
            print(f"  [w{worker_id}] {os.path.basename(pgn_path)} done "
                  f"({positions:,} positions so far)", flush=True)
    return worker_id, games, positions, skipped


def main() -> None:
    file_limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    os.makedirs(OUT_DIR, exist_ok=True)

    pgn_files = sorted(glob.glob(os.path.join(DATA_DIR, "Batch*.pgn")))
    if file_limit is not None:
        pgn_files = pgn_files[:file_limit]
    if not pgn_files:
        print(f"ERROR: no Batch*.pgn in {DATA_DIR}")
        sys.exit(1)

    print("=== Game Data Processor (manifest) ===")
    print(f"Data : {DATA_DIR}")
    print(f"Out  : {OUT_DIR}/pool_games_w*.csv")
    print(f"Files: {len(pgn_files)} | workers {N_WORKERS}\n")

    worker_files: list[list[str]] = [[] for _ in range(N_WORKERS)]
    for i, path in enumerate(pgn_files):
        worker_files[i % N_WORKERS].append(path)
    work = [(wid, files) for wid, files in enumerate(worker_files) if files]

    t0 = time.time()
    with multiprocessing.Pool(processes=len(work)) as pool:
        results = pool.map(worker_fn, work)

    games = sum(r[1] for r in results)
    positions = sum(r[2] for r in results)
    skipped = sum(r[3] for r in results)
    print(f"\n=== Done in {time.time() - t0:.0f}s ===")
    print(f"  {games:,} games | {positions:,} positions | {skipped:,} skipped -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
