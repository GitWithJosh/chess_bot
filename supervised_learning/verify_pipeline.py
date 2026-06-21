"""
Audit the supervised data pipeline for correctness (run from repo root):
    python supervised_learning/verify_pipeline.py

Checks the high-risk invariants:
  1. board encoder is byte-identical to the inference-time Converter,
  2. black-to-move positions are mirrored correctly (no pieces-swapped bug),
  3. move labels round-trip: training index == inference-decoded move,
  4. WDL ordering matches the value head [win, draw, loss].
"""

import sys
import os

import chess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_common as dc
sys.path.insert(0, dc.repo_root())  # so the inference Converter is importable
from reinforcement_learning.helpers.converter import Converter

conv = Converter()
lut = dc.load_move_lut()
idx2uci = {v: k for k, v in lut.items()}
ok_all = True


def check(name, cond):
    global ok_all
    ok_all = ok_all and cond
    print(f"  [{'OK ' if cond else '*** FAIL ***'}] {name}")


print("### 1) ENCODER PARITY vs inference Converter ###")
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
check("friendly king at a1 (chan5)", t[0, 0, 5] == 1 and t[:, :, 5].sum() == 1)
check("enemy king at h8 (chan11)", t[7, 7, 11] == 1 and t[:, :, 11].sum() == 1)
check("channel16==1 everywhere (real side=black)", t[:, :, 16].min() == 1 and t[:, :, 16].max() == 1)
# white Q d4 seen as ENEMY queen by black mover, at vertically-flipped (rank4,file3)
t2 = dc.board_to_input_tensor(chess.Board("8/2k5/8/8/3Q4/8/8/4K3 b - - 0 1"))
check("enemy queen (orig white d4) at mirrored (r4,f3) chan10", t2[4, 3, 10] == 1 and t2[:, :, 10].sum() == 1)

print("\n### 3) MOVE LABEL round-trip (train index -> inference decode) ###")
cases = [
    (chess.STARTING_FEN, "e2e4"), (chess.STARTING_FEN, "g1f3"),
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
    ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8g8"),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", "b8c6"),
    ("4k3/4P3/8/8/8/8/8/4K3 w - - 0 1", "e7e8q"),
    ("4k3/4P3/8/8/8/8/8/4K3 w - - 0 1", "e7e8n"),
    ("4k3/8/8/8/8/8/4p3/4K3 b - - 0 1", "e2e1q"),
    ("4k3/8/8/8/8/8/4p3/4K3 b - - 0 1", "e2e1n"),
]
for fen, mv in cases:
    bd = chess.Board(fen)
    idx = dc.move_to_index(chess.Move.from_uci(mv), bd, lut)
    dec = "<None>" if idx is None else idx2uci[idx]
    if idx is not None and bd.turn == chess.BLACK:
        dec = conv._mirror_move_uci(dec)
    expect = mv[:4] if mv.endswith("q") else mv      # queen promo is stripped by design
    check(f"{mv:6s} idx={str(idx):>5s} -> {dec}", dec == expect)

print("\n### 4) WDL ORDERING vs value head ###")
check("WDL_WIN/DRAW/LOSS == 0/1/2", (dc.WDL_WIN, dc.WDL_DRAW, dc.WDL_LOSS) == (0, 1, 2))
check("game 1-0 white-to-move == win", dc.result_to_wdl_class("1-0", chess.WHITE) == dc.WDL_WIN)
check("game 1-0 black-to-move == loss", dc.result_to_wdl_class("1-0", chess.BLACK) == dc.WDL_LOSS)
check("syzygy +2 == win, -2 == loss, 0 == draw",
      (dc.syzygy_wdl_to_class(2), dc.syzygy_wdl_to_class(-2), dc.syzygy_wdl_to_class(0)) == (0, 2, 1))

print("\n=== ALL CHECKS PASSED ===" if ok_all else "\n=== SOME CHECKS FAILED ===")
sys.exit(0 if ok_all else 1)
