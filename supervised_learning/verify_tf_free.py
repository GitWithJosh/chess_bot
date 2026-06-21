"""TF-free audit of the mirroring/label invariants (no Converter import -> fast).

Inlines the inference-side un-mirror VERBATIM from
reinforcement_learning/helpers/converter.py::_mirror_move_uci so the move
round-trip tests train-label vs inference-decode without loading TensorFlow.
Run: python supervised_learning/verify_tf_free.py
"""
import os
import sys

import chess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_common as dc

lut = dc.load_move_lut()
idx2uci = {v: k for k, v in lut.items()}
ok_all = True


def check(name, cond):
    global ok_all
    ok_all = ok_all and cond
    print(f"  [{'OK ' if cond else '*** FAIL ***'}] {name}")


def mirror_uci(u):  # VERBATIM copy of Converter._mirror_move_uci
    def flip_rank(c):
        return str(9 - int(c))
    r = u[0] + flip_rank(u[1]) + u[2] + flip_rank(u[3])
    if len(u) > 4:
        r += u[4]
    return r


print("### BLACK-to-move SEMANTIC (mirror swaps colour AND flips rank; no swap bug) ###")
# black K a8, white K h1, black to move -> friendly(orig black) K at a1, enemy at h8
t = dc.board_to_input_tensor(chess.Board("k7/8/8/8/8/8/8/7K b - - 0 1"))
check("friendly king at a1 (chan5), exactly one", t[0, 0, 5] == 1 and t[:, :, 5].sum() == 1)
check("enemy king at h8 (chan11), exactly one", t[7, 7, 11] == 1 and t[:, :, 11].sum() == 1)
check("no friendly piece anywhere in enemy planes for that K", t[7, 7, 5] == 0)
check("channel16==1 (real side to move is black)", t[:, :, 16].min() == 1)
# white Q d4 must read as ENEMY queen at vertically-flipped (rank4,file3) for black mover
t2 = dc.board_to_input_tensor(chess.Board("8/2k5/8/8/3Q4/8/8/4K3 b - - 0 1"))
check("enemy queen (orig white d4) at mirrored (r4,f3) chan10", t2[4, 3, 10] == 1 and t2[:, :, 10].sum() == 1)
# white-to-move sanity: same pieces, NOT mirrored, channel16==0
t3 = dc.board_to_input_tensor(chess.Board("8/2k5/8/8/3Q4/8/8/4K3 w - - 0 1"))
check("white-to-move: friendly queen at real d4 (r3,f3) chan4", t3[3, 3, 4] == 1)
check("white-to-move: channel16==0", t3[:, :, 16].max() == 0)

print("\n### MOVE LABEL round-trip (train index -> inference-decoded move) ###")
cases = [
    (chess.STARTING_FEN, "e2e4"), (chess.STARTING_FEN, "g1f3"),
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),    # white O-O
    ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8g8"),    # black O-O
    ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8"),    # black O-O-O
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1", "b8c6"),
    ("4k3/8/8/8/8/6K1/4p3/8 b - - 0 1", "e2e1q"),        # black queen promo (stripped)
    ("4k3/8/8/8/8/6K1/4p3/8 b - - 0 1", "e2e1n"),        # black knight underpromo
    ("4k3/8/8/8/8/6K1/4p3/8 b - - 0 1", "e2e1r"),        # black rook underpromo
    ("8/4P3/8/8/8/8/8/4k1K1 w - - 0 1", "e7e8b"),        # white bishop underpromo
]
for fen, mv in cases:
    bd = chess.Board(fen)
    legal = chess.Move.from_uci(mv) in bd.legal_moves
    idx = dc.move_to_index(chess.Move.from_uci(mv), bd, lut)
    dec = "<None>" if idx is None else idx2uci[idx]
    if idx is not None and bd.turn == chess.BLACK:
        dec = mirror_uci(dec)
    expect = mv[:4] if mv.endswith("q") else mv
    check(f"{mv:6s} legal={legal} idx={str(idx):>5s} -> decoded {dec}", legal and dec == expect)

print("\n### WDL ordering ###")
check("WDL win/draw/loss == 0/1/2", (dc.WDL_WIN, dc.WDL_DRAW, dc.WDL_LOSS) == (0, 1, 2))
check("game 1-0 white==win, black==loss",
      dc.result_to_wdl_class("1-0", chess.WHITE) == 0 and dc.result_to_wdl_class("1-0", chess.BLACK) == 2)
check("game 0-1 black==win, white==loss",
      dc.result_to_wdl_class("0-1", chess.BLACK) == 0 and dc.result_to_wdl_class("0-1", chess.WHITE) == 2)
check("syzygy +2/0/-2 -> win/draw/loss",
      (dc.syzygy_wdl_to_class(2), dc.syzygy_wdl_to_class(0), dc.syzygy_wdl_to_class(-2)) == (0, 1, 2))

print("\n=== ALL PASSED ===" if ok_all else "\n=== FAILURES ABOVE ===")
sys.exit(0 if ok_all else 1)
