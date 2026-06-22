"""
Audit the supervised data pipeline for correctness (run from repo root):
    python supervised_learning/inspect/verify_pipeline.py

Checks the high-risk invariants of dataset_common.py (the shared encode/label
contract every extractor + the assembler depend on):
  1. board encoder is byte-identical to the inference-time Converter,
  2. black-to-move positions are mirrored correctly (colour swap AND rank flip,
     no pieces-swapped bug); white-to-move positions are left as-is,
  3. move labels round-trip: training index == inference-decoded move
     (castling both sides, and every promotion piece),
  4. WDL ordering matches the value head [win, draw, loss].

Only check (1) needs TensorFlow (it imports the real Converter). The pipeline is
deliberately TF-free, so if TF is absent/broken this script SKIPS check (1) with
a warning and still runs 2-4 — the inference-side un-mirror is inlined verbatim
from Converter._mirror_move_uci so the move round-trip never needs the import.
"""

import os
import sys

import chess
import numpy as np

# dataset_common lives in the sibling create_dataset/ package.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "create_dataset"))
import dataset_common as dc
sys.path.insert(0, dc.repo_root())  # so the inference Converter is importable

# Check (1) is the only part that needs TF; everything else stays TF-free.
try:
    from reinforcement_learning.helpers.converter import Converter
    conv = Converter()
except Exception as e:  # TF missing/broken, or Converter import error
    conv = None
    _conv_err = e

lut = dc.load_move_lut()
idx2uci = {v: k for k, v in lut.items()}
ok_all = True


def check(name, cond):
    global ok_all
    ok_all = ok_all and cond
    print(f"  [{'OK ' if cond else '*** FAIL ***'}] {name}")


def mirror_uci(u):  # VERBATIM copy of Converter._mirror_move_uci (keeps this TF-free)
    def flip_rank(c):
        return str(9 - int(c))
    r = u[0] + flip_rank(u[1]) + u[2] + flip_rank(u[3])
    if len(u) > 4:
        r += u[4]
    return r


print("### 1) ENCODER PARITY vs inference Converter ###")
if conv is None:
    print(f"  [SKIP] Converter unavailable ({type(_conv_err).__name__}: {_conv_err})")
    print("         (TF-free run — checks 2-4 below still validate mirror/label logic)")
else:
    fens = [
        chess.STARTING_FEN,
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",   # ep, black to move
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",                          # full castling
        "r3k2r/8/8/8/8/8/8/R3K2R b KQq - 0 5",                           # partial castling, black
        "k7/8/8/8/8/8/8/7K b - - 0 1",
        "5rk1/1p3ppp/pq1Q1b2/8/8/1P3N2/P4PPP/3R2K1 b - - 3 27",
        "4k3/4P3/8/8/8/8/8/4K3 w - - 0 1",
        "8/8/8/8/8/4k3/4p3/4K3 b - - 7 9",
    ]
    maxd = 0.0
    for f in fens:
        a = dc.board_to_input_tensor(chess.Board(f)).astype(np.float32)
        b = conv.board_to_input_tensor(chess.Board(f)).astype(np.float32)
        maxd = max(maxd, float(np.abs(a - b).max()))
    print(f"  max abs diff over {len(fens)} positions = {maxd}")
    check("encoder identical to Converter", maxd == 0.0)

print("\n### 2) BLACK-to-move SEMANTIC (mirror swaps colour AND flips rank) ###")
# black K a8, white K h1, black to move -> friendly(orig black) K at a1; enemy(white) K at h8
t = dc.board_to_input_tensor(chess.Board("k7/8/8/8/8/8/8/7K b - - 0 1"))
check("friendly king at a1 (chan5), exactly one", t[0, 0, 5] == 1 and t[:, :, 5].sum() == 1)
check("enemy king at h8 (chan11), exactly one", t[7, 7, 11] == 1 and t[:, :, 11].sum() == 1)
check("no friendly piece in enemy plane at that square", t[7, 7, 5] == 0)
check("channel16==1 everywhere (real side=black)", t[:, :, 16].min() == 1 and t[:, :, 16].max() == 1)
# white Q d4 seen as ENEMY queen by black mover, at vertically-flipped (rank4,file3)
t2 = dc.board_to_input_tensor(chess.Board("8/2k5/8/8/3Q4/8/8/4K3 b - - 0 1"))
check("enemy queen (orig white d4) at mirrored (r4,f3) chan10", t2[4, 3, 10] == 1 and t2[:, :, 10].sum() == 1)
# white-to-move sanity: same pieces, NOT mirrored, channel16==0
t3 = dc.board_to_input_tensor(chess.Board("8/2k5/8/8/3Q4/8/8/4K3 w - - 0 1"))
check("white-to-move: friendly queen at real d4 (r3,f3) chan4", t3[3, 3, 4] == 1)
check("white-to-move: channel16==0", t3[:, :, 16].max() == 0)

print("\n### 3) MOVE LABEL round-trip (train index -> inference decode) ###")
cases = [
    (chess.STARTING_FEN, "e2e4"), (chess.STARTING_FEN, "g1f3"),
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),    # white O-O
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1"),    # white O-O-O
    ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8g8"),    # black O-O
    ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8"),    # black O-O-O
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", "b8c6"),
    ("8/4P3/8/8/8/8/8/4k1K1 w - - 0 1", "e7e8q"),        # white queen promo (stripped)
    ("8/4P3/8/8/8/8/8/4k1K1 w - - 0 1", "e7e8n"),        # white knight underpromo
    ("8/4P3/8/8/8/8/8/4k1K1 w - - 0 1", "e7e8r"),        # white rook underpromo
    ("8/4P3/8/8/8/8/8/4k1K1 w - - 0 1", "e7e8b"),        # white bishop underpromo
    ("4k3/8/8/8/8/6K1/4p3/8 b - - 0 1", "e2e1q"),        # black queen promo (stripped)
    ("4k3/8/8/8/8/6K1/4p3/8 b - - 0 1", "e2e1n"),        # black knight underpromo
    ("4k3/8/8/8/8/6K1/4p3/8 b - - 0 1", "e2e1r"),        # black rook underpromo
    ("4k3/8/8/8/8/6K1/4p3/8 b - - 0 1", "e2e1b"),        # black bishop underpromo
]
for fen, mv in cases:
    bd = chess.Board(fen)
    legal = chess.Move.from_uci(mv) in bd.legal_moves
    idx = dc.move_to_index(chess.Move.from_uci(mv), bd, lut)
    dec = "<None>" if idx is None else idx2uci[idx]
    if idx is not None and bd.turn == chess.BLACK:
        dec = mirror_uci(dec)
    expect = mv[:4] if mv.endswith("q") else mv      # queen promo is stripped by design
    check(f"{mv:6s} legal={legal} idx={str(idx):>5s} -> {dec}", legal and dec == expect)

print("\n### 4) WDL ORDERING vs value head ###")
check("WDL_WIN/DRAW/LOSS == 0/1/2", (dc.WDL_WIN, dc.WDL_DRAW, dc.WDL_LOSS) == (0, 1, 2))
check("game 1-0 white-to-move == win", dc.result_to_wdl_class("1-0", chess.WHITE) == dc.WDL_WIN)
check("game 1-0 black-to-move == loss", dc.result_to_wdl_class("1-0", chess.BLACK) == dc.WDL_LOSS)
check("game 0-1 black-to-move == win", dc.result_to_wdl_class("0-1", chess.BLACK) == dc.WDL_WIN)
check("game 0-1 white-to-move == loss", dc.result_to_wdl_class("0-1", chess.WHITE) == dc.WDL_LOSS)
check("syzygy +2 == win, -2 == loss, 0 == draw",
      (dc.syzygy_wdl_to_class(2), dc.syzygy_wdl_to_class(-2), dc.syzygy_wdl_to_class(0)) == (0, 2, 1))

print("\n=== ALL CHECKS PASSED ===" if ok_all else "\n=== SOME CHECKS FAILED ===")
sys.exit(0 if ok_all else 1)
