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
inference-time Converter, so the
network sees a consistent encoding at train and play time.
"""

import hashlib
import os
import sys

import chess
import numpy as np

# WDL class indices — MUST match the value head's [win, draw, loss] one-hot.
WDL_WIN, WDL_DRAW, WDL_LOSS = 0, 1, 2
N_WDL = 3
POLICY_SIZE = 1858

# --- Stockfish soft value targets (2026-07) --------------------------------
# The game-outcome label is noisy by construction: every position of a won game
# is labeled "win" even when it was objectively balanced (measured on a 12k
# sample, Stockfish d12 calls 48% of positions from won games drawn). So the
# value target is BLENDED with Stockfish's WDL for that position:
#
#     target = (1 - VALUE_LAMBDA) * onehot(outcome)  +  VALUE_LAMBDA * sf_wdl
#
# Balancing still runs on the OUTCOME label (see build_dataset.plan), so chunk
# composition — and the chunk count — is identical to the outcome-only build;
# only the target inside each row gets richer. Tablebase rows are NOT annotated:
# Syzygy is exact ground truth and d12 Stockfish would only degrade it.
VALUE_LAMBDA = 0.5

SF_SHARDS = 64           # cache is sharded by fen hash so writes stay append-only
SF_MATE_CP = 32000       # python-chess mate_score; cp is clipped to +-this

# Per-sample source ids stored in each chunk's "sources" uint8 array (written
# by build_dataset.py). Enables per-source val metrics in training. Defender-
# side puzzle rows count as "puzzle" and drawn tablebase rows as "tablebase" —
# the source describes where the position came from, not its label. MUST stay
# in sync with the SOURCE_NAMES copy in train_supervised.py (duplicated there
# so the training script stays standalone on Kaggle).
SOURCE_IDS = {"game": 0, "puzzle": 1, "tablebase": 2}

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

# Stockfish annotation cache: sf_XX.csv shards of "fen,cp,w,d,l" (w/d/l permille,
# side-to-move POV). Append-only, so an interrupted overnight run resumes by
# simply skipping the FENs already present.
SF_CACHE_DIR = os.path.join(_ROOT, "supervised_learning", "sf_cache")


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


# ---------------------------------------------------------------------------
# Stockfish annotation cache
# ---------------------------------------------------------------------------

def fen_hash(fen: str) -> int:
    """Stable 64-bit hash of a FEN. Used to dedupe the annotation work list and
    to index the cache without keeping 15M FEN strings in memory (the uint64
    keys are ~120 MB; the strings would be >1 GB). Collision probability over
    15M positions is ~6e-6, i.e. irrelevant at this scale.

    Must be STABLE across processes, so hashlib — not Python's hash(), which is
    randomised per interpreter by PYTHONHASHSEED.
    """
    return int.from_bytes(hashlib.blake2b(fen.encode(), digest_size=8).digest(), "little")


def sf_shard_path(shard: int) -> str:
    return os.path.join(SF_CACHE_DIR, f"sf_{shard:02d}.csv")


def load_sf_cache(verbose: bool = True):
    """All cached annotations -> (sorted uint64 keys, uint16 wdl permille (n,3)).

    Returned sorted so callers look up with np.searchsorted. Malformed trailing
    lines (possible if the process was killed mid-write) are skipped rather than
    raising — the append-only shards are meant to survive a hard stop.
    """
    keys: list[np.ndarray] = []
    wdls: list[np.ndarray] = []
    if not os.path.isdir(SF_CACHE_DIR):
        return np.empty(0, np.uint64), np.empty((0, N_WDL), np.uint16)

    for shard in range(SF_SHARDS):
        path = sf_shard_path(shard)
        if not os.path.exists(path):
            continue
        k_s, w_s = [], []
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    fen, _cp, w, d, l = line.rstrip("\n").rsplit(",", 4)
                    row = (int(w), int(d), int(l))
                except ValueError:
                    continue                      # truncated final line
                k_s.append(fen_hash(fen))
                w_s.append(row)
        if k_s:
            keys.append(np.array(k_s, dtype=np.uint64))
            wdls.append(np.array(w_s, dtype=np.uint16))

    if not keys:
        return np.empty(0, np.uint64), np.empty((0, N_WDL), np.uint16)

    k = np.concatenate(keys)
    w = np.concatenate(wdls)
    order = np.argsort(k, kind="stable")
    k, w = k[order], w[order]
    uniq = np.concatenate(([True], k[1:] != k[:-1]))   # last writer wins is fine
    k, w = k[uniq], w[uniq]
    if verbose:
        print(f"  SF cache: {len(k):,} annotated positions", flush=True)
    return k, w


def blend_value(wdl_class: int, sf_wdl, lam: float = VALUE_LAMBDA) -> np.ndarray:
    """Value target: outcome one-hot blended with Stockfish's WDL distribution.

    sf_wdl is a (3,) permille triple in the side-to-move frame, or None for rows
    with no annotation (tablebase, or a game/puzzle row Stockfish never reached),
    which fall back to the pure one-hot label.
    """
    value = np.zeros(N_WDL, dtype=np.float32)
    value[wdl_class] = 1.0
    if sf_wdl is None:
        return value
    sf = np.asarray(sf_wdl, dtype=np.float32)
    sf = sf / max(float(sf.sum()), 1e-9)
    return (1.0 - lam) * value + lam * sf


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


def encode_sample(fen: str, move_idx: int, wdl: int, sf_wdl=None,
                  lam: float = VALUE_LAMBDA):
    """One manifest row -> (board (8,8,20) f16, policy (1858,) f32, value (3,) f32).

    move_idx -1 marks a VALUE-ONLY row (drawn tablebase positions, which have
    no single best move): the policy target stays all-zero, whose categorical
    cross-entropy loss and gradient are exactly 0. train_supervised.py also
    derives per-sample policy weights from policies.sum(axis=1), so these rows
    are excluded from the policy accuracy metric as well.

    sf_wdl (permille triple, side-to-move POV) turns the value target into a
    SOFT one — see blend_value. Passing None reproduces the old one-hot target
    exactly, so the outcome-only build is still reachable.
    """
    boards = board_to_input_tensor(chess.Board(fen))
    policy = np.zeros(POLICY_SIZE, dtype=np.float32)
    if move_idx >= 0:
        policy[move_idx] = 1.0
    value = blend_value(wdl, sf_wdl, lam)
    return boards, policy, value
