"""
Per-phase Average Centipawn Loss (ACPL) for the moves in one or more PGNs.

Winrate tells you which net is stronger overall; it can't tell you *where*. This
replays every game move-by-move, asks Stockfish to evaluate each position, and
charges each move the centipawn drop it caused versus the prior position. Losses
are bucketed by game phase (opening / middlegame / endgame) and by which player
made the move, so you can see e.g. "endgame ACPL fell from 180 to 95" after the
endgame-heavy data change — a signal independent of opponent or who won.

Phase is by material (the tapered-eval phase weight: N/B=1, R=2, Q=4; 24 at the
start), not move number, so a position is classified the same however it arose.

Centipawn loss: with e_i = Stockfish's white-POV eval of the position before
ply i, a white move's loss is e_i - e_{i+1} and a black move's is e_{i+1} - e_i,
clamped at >= 0. Each game position is evaluated exactly once.

TF-free: needs only python-chess and a Stockfish binary.

Edit the CONFIG block below, then just run the file (no arguments):
    python supervised_learning/phase_acpl.py
"""

import glob
import os
import shutil

import chess
import chess.engine
import chess.pgn

# ============================ CONFIG — edit, then run ============================
PGNS = ["supervised_learning/results/match_*_mcts.pgn"]  # path(s) or glob(s)
DEPTH = 12          # Stockfish search depth per position
STOCKFISH = None    # None = auto-detect (PATH, $STOCKFISH_PATH, C:\stockfish\stockfish.exe)
MAX_GAMES = None    # cap games per file (None = all); set small to dry-run
THREADS = 1         # Stockfish Threads option
HASH_MB = 128       # Stockfish Hash (MB)
# ================================================================================

PHASE_WEIGHTS = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
PHASE_START = 24
MATE_CP = 10_000  # cap for mate scores when converting to centipawns

STOCKFISH_FALLBACKS = (
    os.environ.get("STOCKFISH_PATH"),
    r"C:\stockfish\stockfish.exe",
    "/usr/bin/stockfish",
    "/usr/local/bin/stockfish",
)


def find_stockfish(explicit: str | None) -> str:
    for cand in (explicit, shutil.which("stockfish"), *STOCKFISH_FALLBACKS):
        if cand and os.path.exists(cand):
            return cand
        if cand and shutil.which(cand):  # bare name on PATH
            return cand
    raise FileNotFoundError(
        "Stockfish not found. Put it on PATH, set STOCKFISH_PATH, or pass "
        "--stockfish C:\\path\\to\\stockfish.exe"
    )


def game_phase(board: chess.Board) -> str:
    total = 0
    for piece_type, weight in PHASE_WEIGHTS.items():
        total += weight * len(board.pieces(piece_type, chess.WHITE))
        total += weight * len(board.pieces(piece_type, chess.BLACK))
    if total >= 20:
        return "opening"
    if total >= 8:
        return "middlegame"
    return "endgame"


def eval_white_cp(engine: chess.engine.SimpleEngine, board: chess.Board, depth: int) -> int:
    """White-POV centipawn eval of `board`. Terminal positions resolved exactly."""
    if board.is_checkmate():
        # side to move is mated -> they lost
        return -MATE_CP if board.turn == chess.WHITE else MATE_CP
    if board.is_game_over():  # stalemate / insufficient material / 75-move / etc.
        return 0
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    return info["score"].white().score(mate_score=MATE_CP)


def analyze_game(game: chess.pgn.Game, engine, depth: int, stats: dict):
    moves = list(game.mainline_moves())
    if not moves:
        return
    white_name = game.headers.get("White", "White")
    black_name = game.headers.get("Black", "Black")

    board = game.board()
    # evals[i] = white-POV eval of the position BEFORE ply i; one extra for the final position
    evals = []
    for mv in moves:
        evals.append(eval_white_cp(engine, board, depth))
        board.push(mv)
    evals.append(eval_white_cp(engine, board, depth))

    board = game.board()
    for ply, mv in enumerate(moves):
        delta = evals[ply] - evals[ply + 1]  # white's loss this ply (white POV)
        mover_is_white = board.turn == chess.WHITE
        cpl = max(0, delta if mover_is_white else -delta)
        name = white_name if mover_is_white else black_name
        phase = game_phase(board)
        bucket = stats.setdefault(name, {})
        s, c = bucket.get(phase, (0, 0))
        bucket[phase] = (s + cpl, c + 1)
        board.push(mv)


def print_report(stats: dict):
    phases = ["opening", "middlegame", "endgame"]
    name_w = max((len(n) for n in stats), default=6)
    name_w = max(name_w, 6)
    header = f"{'player':<{name_w}} | " + " | ".join(f"{p:^18}" for p in phases)
    print("\n=== ACPL by phase (lower = better; 'n' = moves scored) ===")
    print(header)
    print("-" * len(header))
    for name in sorted(stats):
        cells = []
        for p in phases:
            s, c = stats[name].get(p, (0, 0))
            cells.append(f"{(s / c):6.1f} (n={c:4d})" if c else f"{'-':>13}")
        print(f"{name:<{name_w}} | " + " | ".join(f"{cell:^18}" for cell in cells))


def main():
    paths = [p for pat in PGNS for p in sorted(glob.glob(pat))]
    if not paths:
        raise SystemExit(f"no PGNs matched: {PGNS}")

    sf_path = find_stockfish(STOCKFISH)
    print(f"Stockfish: {sf_path}  (depth {DEPTH})")
    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    engine.configure({"Threads": THREADS, "Hash": HASH_MB})

    stats: dict[str, dict[str, tuple[int, int]]] = {}
    try:
        for path in paths:
            print(f"analyzing {path} ...", flush=True)
            with open(path) as f:
                seen = 0
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    analyze_game(game, engine, DEPTH, stats)
                    seen += 1
                    if MAX_GAMES and seen >= MAX_GAMES:
                        break
                print(f"  {seen} games")
    finally:
        engine.quit()

    print_report(stats)


if __name__ == "__main__":
    main()
