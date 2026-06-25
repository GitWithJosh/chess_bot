"""AlphaZero/Leela 73-plane policy move encoding.

The policy is a stack of 73 planes per board square (8x8x73 = 4672 cells), of
which exactly 1858 cells correspond to the moves in ``move_lookup.json``. Each
move is encoded as a ``(from_square, plane)`` pair:

    planes  0-55 : "queen"/sliding moves  — 8 compass directions x 7 distances
    planes 56-63 : knight moves           — 8 fixed (drank, dfile) offsets
    planes 64-72 : under-promotions       — 3 file-directions x 3 pieces (N,B,R)

Queen promotions are *not* in the under-promotion planes: they share the plain
sliding-move plane (matching ``move_lookup`` keys, which strip the 'q' suffix).

All moves are in the side-to-move ("friendly", white-at-the-bottom) frame that
the Converter already produces, so pawns always advance toward rank 8
(d_rank > 0) and the gather is spatially aligned with the board input tensor.

The exact ordering of directions/offsets within each group is arbitrary (the
network learns whatever each plane means); it only has to be fixed and
collision-free. ``build_gather_indices`` enforces that invariant.
"""

import json

import numpy as np

BOARD = 8
NUM_PLANES = 73

# 8 sliding directions as (d_rank, d_file): N, NE, E, SE, S, SW, W, NW.
_QUEEN_DIRS = [
    (1, 0),    # N
    (1, 1),    # NE
    (0, 1),    # E
    (-1, 1),   # SE
    (-1, 0),   # S
    (-1, -1),  # SW
    (0, -1),   # W
    (1, -1),   # NW
]

# 8 knight offsets as (d_rank, d_file) — all four (+-2,+-1) and (+-1,+-2).
_KNIGHT_OFFSETS = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1),
]

# Under-promotion: file-delta -> direction slot, suffix -> piece slot.
_UNDERPROMO_DIRS = {-1: 0, 0: 1, 1: 2}
_UNDERPROMO_PIECES = {"n": 0, "b": 1, "r": 2}

_QUEEN_BASE = 0
_KNIGHT_BASE = 56
_UNDERPROMO_BASE = 64


def _parse_square(sq: str) -> tuple[int, int]:
    """'e4' -> (rank, file), zero-indexed (rank 0 == rank 1, file 0 == a-file)."""
    file = ord(sq[0]) - ord("a")
    rank = int(sq[1]) - 1
    return rank, file


def move_uci_to_plane(uci: str) -> int:
    """Map a friendly-perspective lookup UCI to its policy plane in ``[0, 73)``.

    Expects ``move_lookup`` keys: 4-char moves (incl. queen promotions, with the
    'q' already stripped) or 5-char under-promotions ending in n/b/r.
    """
    from_rank, from_file = _parse_square(uci[0:2])
    to_rank, to_file = _parse_square(uci[2:4])
    dr = to_rank - from_rank
    df = to_file - from_file

    # --- Under-promotion (knight / bishop / rook): 9 planes ---
    if len(uci) == 5:
        piece = uci[4]
        if piece not in _UNDERPROMO_PIECES:
            raise ValueError(f"unexpected promotion piece in {uci!r}")
        if df not in _UNDERPROMO_DIRS:
            raise ValueError(f"under-promotion {uci!r} has bad file delta {df}")
        return _UNDERPROMO_BASE + 3 * _UNDERPROMO_DIRS[df] + _UNDERPROMO_PIECES[piece]

    # --- Knight move: 8 planes ---
    if (abs(dr), abs(df)) in ((1, 2), (2, 1)):
        return _KNIGHT_BASE + _KNIGHT_OFFSETS.index((dr, df))

    # --- Sliding move (queen/rook/bishop/king, pawn pushes & captures, queen
    #     promotions, castling-as-2-square-king-move): 56 planes ---
    dist = max(abs(dr), abs(df))
    if dist == 0:
        raise ValueError(f"null move {uci!r}")
    step = (dr // dist, df // dist)
    if step not in _QUEEN_DIRS or step[0] * dist != dr or step[1] * dist != df:
        raise ValueError(f"move {uci!r} is neither a knight nor an axis-aligned move")
    return _QUEEN_BASE + _QUEEN_DIRS.index(step) * 7 + (dist - 1)


def build_gather_indices(lookup: dict, num_planes: int = NUM_PLANES) -> np.ndarray:
    """Flat gather indices into a flattened ``(8, 8, num_planes)`` policy tensor.

    Returns an ``int32`` array of length ``len(lookup)`` where entry ``i`` is the
    flat index of move ``i``'s cell, using row-major (channels-last) order:

        flat = from_rank * (8 * num_planes) + from_file * num_planes + plane

    Raises ``ValueError`` if any plane is out of range or two moves map to the
    same cell (the mapping must be a bijection onto its image).
    """
    num_moves = len(lookup)
    indices = np.full(num_moves, -1, dtype=np.int32)
    seen: dict[int, str] = {}
    for key, uci in lookup.items():
        i = int(key)
        plane = move_uci_to_plane(uci)
        if not 0 <= plane < num_planes:
            raise ValueError(f"plane {plane} out of range for move {uci!r}")
        from_rank, from_file = _parse_square(uci[0:2])
        flat = from_rank * (BOARD * num_planes) + from_file * num_planes + plane
        if flat in seen:
            raise ValueError(
                f"collision: {seen[flat]!r} and {uci!r} both map to "
                f"(rank={from_rank}, file={from_file}, plane={plane})"
            )
        seen[flat] = uci
        indices[i] = flat

    if (indices < 0).any():
        missing = np.where(indices < 0)[0]
        raise ValueError(f"lookup keys are not a contiguous 0..N-1 range; missing {missing}")
    return indices


def gather_indices_from_lookup_path(
    path: str = "reinforcement_learning/move_lookup.json",
    num_planes: int = NUM_PLANES,
) -> np.ndarray:
    """Load ``move_lookup.json`` and build the gather index array from it."""
    with open(path, "r") as f:
        lookup = json.load(f)
    return build_gather_indices(lookup, num_planes)