"""
Syzygy endgame tablebase -> mate-sequence manifest (fen,move_idx,wdl rows).

Run from the chess_bot root:
    python supervised_learning/create_dataset/process_tablebase_data.py        # full run
    python supervised_learning/create_dataset/process_tablebase_data.py 20000  # test: ~20k positions

Idea (the "sense of goal" data): pick a decisive <=5-piece material config, set
up a random legal position, then play the tablebase-optimal line and record
every position of it — BUT only if the line actually reaches checkmate:

  - the winning side to move -> WIN, policy = win-preserving move nearest to
    zeroing (fastest progress; ties broken toward unvisited positions so the
    line can't shuffle/repeat), with immediate mate always preferred,
  - the losing side to move  -> LOSS, policy = longest-resistance move (max DTZ).

A walk that fails to mate (DTZ is distance-to-zeroing, not to mate, so a few
lines stall) is discarded entirely, so no shuffling moves leak into the policy
targets. The position colours are also flipped 50% of the time so the strong
side is White/Black equally often. Draws are intentionally skipped — they're
abundant in the game data; what's scarce is this win/loss "drive to mate" signal.

Output: supervised_learning/pools/pool_tablebase_w{worker}.csv
"""

import multiprocessing
import os
import random
import sys
import time

import chess
import chess.syzygy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_common as dc  # noqa: E402

TB_DIR   = os.path.join(dc.repo_root(), "supervised_learning", "supervised_data", "syzygy_3-4-5")
OUT_DIR  = os.path.join(dc.repo_root(), "supervised_learning", "pools")

TARGET_POSITIONS = 1_000_000   # assembler needs ~5%/chunk; 1M covers ~400 chunks with margin
N_WORKERS        = 12
MAX_PLIES        = 250         # safety cap per walk (DTZ lines run longer than shortest mate)
SEED             = 1234

# Decisive material configs to start walks from: (white_extra, black_extra) added
# to the two kings. Covers the fundamental mates the resignation data starves the
# net of, plus common K+piece-vs-K+piece wins. Side to move is randomised, so
# e.g. ("Q","R") also yields the R-vs-Q losing side's positions.
MATERIALS = [
    ("Q", ""), ("R", ""), ("P", ""),
    ("QQ", ""), ("QR", ""), ("RR", ""), ("BB", ""), ("BN", ""), ("QP", ""), ("RP", ""), ("PP", ""),
    ("Q", "R"), ("Q", "B"), ("Q", "N"), ("Q", "P"),
    ("R", "B"), ("R", "N"), ("R", "P"),
    ("QR", "Q"), ("RR", "R"), ("QP", "Q"), ("RP", "R"),
]

_PIECE = {"Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP, "N": chess.KNIGHT, "P": chess.PAWN}


def _random_position(rng: random.Random, material: tuple[str, str]) -> chess.Board | None:
    """A random LEGAL board with the given extra material, or None if a few tries fail."""
    white_extra, black_extra = material
    n_pieces = 2 + len(white_extra) + len(black_extra)
    for _ in range(40):
        board = chess.Board.empty()
        sqs = rng.sample(range(64), n_pieces)
        board.set_piece_at(sqs[0], chess.Piece(chess.KING, chess.WHITE))
        board.set_piece_at(sqs[1], chess.Piece(chess.KING, chess.BLACK))
        i = 2
        for letter in white_extra:
            board.set_piece_at(sqs[i], chess.Piece(_PIECE[letter], chess.WHITE)); i += 1
        for letter in black_extra:
            board.set_piece_at(sqs[i], chess.Piece(_PIECE[letter], chess.BLACK)); i += 1
        board.turn = rng.choice([chess.WHITE, chess.BLACK])
        # is_valid() rejects adjacent kings, the side-not-to-move being in check,
        # and pawns on the back ranks — exactly the illegal setups we must avoid.
        if board.is_valid() and not board.is_game_over():
            # 50% colour swap so the strong side is White / Black equally often.
            # MATERIALS always gives the extra material to White, so without this
            # every win is real-White-to-move and every loss real-Black-to-move —
            # letting the net read the outcome off channel 16 (side to move) rather
            # than the pieces. apply_mirror() flips colours, board and side to move
            # together, so the side-to-move's WDL is unchanged. The board encoder
            # then mirrors to the mover's frame, so only channel 16 differs.
            if rng.random() < 0.5:
                board.apply_mirror()
            return board
    return None


def _choose_move(tb: chess.syzygy.Tablebase, board: chess.Board, stm_wdl: int, seen: set):
    """The tablebase-optimal move to RECORD as the policy target for this position.

    `seen` holds the EPDs already visited this walk. DTZ is distance-to-*zeroing*,
    not distance-to-mate, so plain DTZ-minimising play can plateau and draw a won
    game by repetition. So the winning side prefers a win-preserving move into an
    UNVISITED position (breaking ties by smallest DTZ); that guarantees progress
    to mate. The losing side just resists longest (largest DTZ).
    """
    win_cands = []   # (|dtz|, repeats: bool, move)
    loss_cands = []  # (|dtz|, move)
    for m in board.legal_moves:
        board.push(m)
        if board.is_checkmate():
            board.pop()
            return m  # nothing beats immediate mate
        child_wdl = tb.probe_wdl(board)        # opponent's POV after our move
        child_dtz = tb.probe_dtz(board)
        child_epd = board.epd()
        board.pop()
        our_wdl = -child_wdl                   # our result after playing m
        if stm_wdl >= 2:
            if our_wdl >= 2:                   # keep the win
                win_cands.append((abs(child_dtz), child_epd in seen, m))
        elif stm_wdl <= -2:
            loss_cands.append((abs(child_dtz), m))

    if stm_wdl >= 2 and win_cands:
        # Nearest to zeroing FIRST (true DTZ-optimal progress); unvisited only as a
        # tie-break, to break the DTZ plateaus that would otherwise shuffle/repeat.
        win_cands.sort(key=lambda c: (c[0], c[1]))
        return win_cands[0][2]
    if stm_wdl <= -2 and loss_cands:
        loss_cands.sort(key=lambda c: -c[0])         # longest resistance
        return loss_cands[0][1]
    return next(iter(board.legal_moves))             # fallback (shouldn't trigger)


def _walk(tb, board, lut, out) -> int:
    """Play the optimal line; emit the positions ONLY if it reaches checkmate.

    Buffering and discarding non-mating lines means every recorded move provably
    belongs to a completed mating sequence — no shuffling/repetition moves leak
    into the policy targets. Per-position WDL labels are exact regardless.
    """
    rows = []
    seen = {board.epd()}
    for _ in range(MAX_PLIES):
        if board.is_game_over() or chess.popcount(board.occupied) > 5:
            break
        stm_wdl = tb.probe_wdl(board)
        if stm_wdl == 0:        # drifted into a drawn position — abandon, we want decisive
            break
        move = _choose_move(tb, board, stm_wdl, seen)
        idx = dc.move_to_index(move, board, lut)
        if idx is not None:
            rows.append(dc.manifest_line(board.fen(), idx, dc.syzygy_wdl_to_class(stm_wdl)))
        board.push(move)
        seen.add(board.epd())

    if board.is_checkmate():
        out.writelines(rows)
        return len(rows)
    return 0


def worker_fn(args) -> tuple[int, int]:
    worker_id, target, seed = args
    rng = random.Random(seed)
    lut = dc.load_move_lut()
    tb = chess.syzygy.open_tablebase(TB_DIR)
    out_path = os.path.join(OUT_DIR, f"pool_tablebase_w{worker_id}.csv")
    written = 0
    with open(out_path, "w", encoding="utf-8") as out:
        while written < target:
            board = _random_position(rng, rng.choice(MATERIALS))
            if board is None:
                continue
            if tb.probe_wdl(board) == 0:     # skip drawn starts; we want mate sequences
                continue
            written += _walk(tb, board, lut, out)
    tb.close()
    return worker_id, written


def main() -> None:
    target_total = int(sys.argv[1]) if len(sys.argv) > 1 else TARGET_POSITIONS
    os.makedirs(OUT_DIR, exist_ok=True)
    per = max(1, target_total // N_WORKERS)

    print("=== Tablebase Data Processor (mate-sequence walks) ===")
    print(f"TB dir : {TB_DIR}")
    print(f"Out    : {OUT_DIR}/pool_tablebase_w*.csv")
    print(f"Target : {target_total:,} positions across {N_WORKERS} workers ({per:,} each)\n")

    t0 = time.time()
    work = [(w, per, SEED + w) for w in range(N_WORKERS)]
    with multiprocessing.Pool(processes=N_WORKERS) as pool:
        results = pool.map(worker_fn, work)

    total = sum(r[1] for r in results)
    print(f"=== Done in {time.time() - t0:.0f}s — {total:,} positions across {len(results)} shards ===")


if __name__ == "__main__":
    main()
