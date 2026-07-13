"""
Shared helpers for the supervised data pipeline (games + puzzles + tablebase).

All three extractors emit the SAME compact manifest row:

    fen,move_idx,wdl

  fen      : the position, side-to-move's turn (a full FEN, no commas so CSV-safe)
  move_idx : index into the 1858-move policy vector, ALREADY mirrored to the
             side-to-move frame (see move_to_index) — i.e. the policy target
  wdl      : 0 = win, 1 = draw, 2 = loss, from the side-to-move's perspective

The assembler (build_dataset.py) later turns each row into the tensors
train_supervised.py expects:
    boards   float16 (8,8,20)   via Converter.board_to_input_tensor
    policies float32 (1858,)     one-hot at move_idx
    values   float32 (3,)        one-hot [win, draw, loss] at wdl

Keeping move-indexing, board encoding and the WDL convention in ONE place is
what guarantees game/puzzle/TB samples encode byte-identically to the
inference-time Converter (verified by inspect/verify_pipeline.py), so the
network sees a consistent encoding at train and play time.
"""

import os
import sys

import chess
import numpy as np

# WDL class indices — MUST match the value head's [win, draw, loss] one-hot.
WDL_WIN, WDL_DRAW, WDL_LOSS = 0, 1, 2
N_WDL = 3
POLICY_SIZE = 1858

# Per-sample source ids stored in each chunk's "sources" uint8 array (written
# by build_dataset.py). Enables per-source val metrics and, later, per-source
# loss weighting in training. 3/4 are reserved for planned data additions
# (puzzle defender-side rows, drawn tablebase rows). MUST stay in sync with
# the SOURCE_NAMES copy in train_supervised.py (duplicated there so the
# training script stays standalone on Kaggle).
SOURCE_IDS = {"game": 0, "puzzle": 1, "tablebase": 2, "puzzle_def": 3, "tb_draw": 4}

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def repo_root() -> str:
    return _ROOT


def ensure_on_path() -> None:
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)


def load_move_lut() -> dict[str, int]:
    """uci -> policy index, in the side-to-move frame (queen promotions stripped).

    Reads reinforcement_learning/move_lookup.json directly (index->uci) and
    inverts it. Deliberately does NOT import Converter, which pulls in
    TensorFlow — the data pipeline is pure numpy/chess so it parallelises across
    many worker processes without each loading TF (which exhausts memory).
    """
    import json
    path = os.path.join(_ROOT, "reinforcement_learning", "move_lookup.json")
    with open(path) as f:
        lookup = json.load(f)  # {index_str: uci}
    return {uci: int(idx) for idx, uci in lookup.items()}


def move_to_index(move: chess.Move, board: chess.Board, lut: dict[str, int]) -> int | None:
    """Policy index for `move` played on `board`, in the side-to-move frame.

    Identical logic to process_pgn_data._move_to_index: when black is to move the
    move is mirrored (ranks flipped) so the network always sees the mover as
    "white", and the default queen promotion suffix is dropped (not in the LUT).
    Returns None if the move isn't representable (rare; caller should skip).
    """
    uci = move.uci()
    if board.turn == chess.BLACK:
        uci = uci[0] + str(9 - int(uci[1])) + uci[2] + str(9 - int(uci[3])) + uci[4:]
    if move.promotion == chess.QUEEN:
        uci = uci[:4]
    return lut.get(uci)


def result_to_wdl_class(result: str, turn: chess.Color) -> int | None:
    """Game result string -> WDL class from the side-to-move's perspective."""
    if result == "1/2-1/2":
        return WDL_DRAW
    if result == "1-0":
        return WDL_WIN if turn == chess.WHITE else WDL_LOSS
    if result == "0-1":
        return WDL_WIN if turn == chess.BLACK else WDL_LOSS
    return None


def syzygy_wdl_to_class(wdl: int) -> int:
    """Syzygy probe_wdl() value (+2/+1 win, 0 draw, -1/-2 loss) -> WDL class.

    +1/-1 (cursed win / blessed loss = win/loss only outside the 50-move rule)
    are treated as draw, which is the honest label for a position the side to
    move cannot actually convert under the rules.
    """
    if wdl >= 2:
        return WDL_WIN
    if wdl <= -2:
        return WDL_LOSS
    return WDL_DRAW


# ---------------------------------------------------------------------------
# Manifest IO — one CSV-ish line per position. FENs contain no commas, so a
# plain comma split yields exactly [fen, move_idx, wdl]; no quoting needed.
# ---------------------------------------------------------------------------

def manifest_line(fen: str, move_idx: int, wdl: int) -> str:
    return f"{fen},{move_idx},{wdl}\n"


def parse_manifest_line(line: str) -> tuple[str, int, int]:
    fen, mi, w = line.rstrip("\n").rsplit(",", 2)
    return fen, int(mi), int(w)


def board_to_input_tensor(board: chess.Board) -> np.ndarray:
    """(8,8,20) position-only tensor — byte-identical to Converter.board_to_input_tensor.

    Duplicated here (rather than imported) so the data pipeline stays TF-free; see
    load_move_lut. If the Converter encoding ever changes, change it here too.
    """
    oriented_board = board.mirror() if board.turn == chess.BLACK else board.copy()
    tensor = np.zeros((8, 8, 20), dtype=np.float16)
    friendly_color = chess.WHITE  # after orientation the side to move is always white

    for square, piece in oriented_board.piece_map().items():
        rank, file = square // 8, square % 8
        ptype = piece.piece_type - 1
        if piece.color == friendly_color:
            tensor[rank, file, ptype] = 1
        else:
            tensor[rank, file, 6 + ptype] = 1

    tensor[:, :, 12] = float(oriented_board.has_queenside_castling_rights(chess.WHITE))
    tensor[:, :, 13] = float(oriented_board.has_kingside_castling_rights(chess.WHITE))
    tensor[:, :, 14] = float(oriented_board.has_queenside_castling_rights(chess.BLACK))
    tensor[:, :, 15] = float(oriented_board.has_kingside_castling_rights(chess.BLACK))
    tensor[:, :, 16] = float(board.turn == chess.BLACK)
    tensor[:, :, 17] = board.halfmove_clock / 100
    if oriented_board.has_legal_en_passant():
        ep = oriented_board.ep_square
        tensor[ep // 8, ep % 8, 18] = 1
    tensor[:, :, 19] = 1
    return tensor


def encode_sample(fen: str, move_idx: int, wdl: int):
    """One manifest row -> (board (8,8,20) f16, policy (1858,) f32, value (3,) f32)."""
    boards = board_to_input_tensor(chess.Board(fen))
    policy = np.zeros(POLICY_SIZE, dtype=np.float32)
    policy[move_idx] = 1.0
    value = np.zeros(N_WDL, dtype=np.float32)
    value[wdl] = 1.0
    return boards, policy, value
