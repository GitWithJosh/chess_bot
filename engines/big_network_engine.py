"""BigNetwork engine — plays via a trained supervised-learning checkpoint.

Greedy policy inference only (no MCTS): one forward pass per move, then pick the
highest-probability legal move. Bridges the GUI's custom board objects to the
python-chess + Converter stack the network was trained against.

Given the game's move_history, we replay it into a python-chess board so the
converter can fill the network's 7-move history planes (channels 13-103). Without
history (or from a non-standard start) we fall back to a stackless board built
from the FEN — correct position, just empty history planes.
"""

import os
import sys

# big_network.py imports from networks/, monte_carlo_tree_search/, helpers/ —
# all under reinforcement_learning/. Make both the repo root and that package
# dir importable before pulling anything in.
_here = os.path.dirname(os.path.abspath(__file__))   # chess_bot/engines/
_root = os.path.dirname(_here)                        # chess_bot/
_rl   = os.path.join(_root, "reinforcement_learning")
for _p in (_root, _rl):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import chess

from board.board import BoardState
from move_generation.move import Move
from engines.engine import ChessEngine
from utils.coordinates import algebraic_to_indices

from networks.big_network import BigNetwork
from helpers.converter import Converter


_PROMO_WORD = {"q": "queen", "r": "rook", "b": "bishop", "n": "knight"}
_PROMO_CHAR = {"queen": "q", "rook": "r", "bishop": "b", "knight": "n"}


def _custom_move_to_uci(move: Move) -> str:
    """Custom Move -> UCI string (e.g. e2e4, e7e8q, e1g1 for castling)."""
    uci = move.from_square + move.to_square
    if move.promotion_piece:
        uci += _PROMO_CHAR.get(move.promotion_piece, "")
    return uci


class BigNetworkEngine(ChessEngine):
    """Wraps a trained BigNetwork checkpoint as a playable engine."""

    def __init__(self, weights_path: str):
        self._net = BigNetwork()
        self._net.load(weights_path)
        self._converter = Converter()   # opens reinforcement_learning/move_lookup.json (CWD-relative)
        self._weights_path = weights_path

    def get_best_move(
        self, board_state: BoardState, move_history: list[Move] | None = None
    ) -> Move | None:
        chess_board = self._board_with_history(board_state, move_history)
        if chess_board.is_game_over():
            return None

        # Converter masks illegal moves against this board and mirrors for Black.
        self._converter.board = chess_board

        # Converter is annotated tf.Tensor but builds/consumes numpy at runtime,
        # so its signatures disagree with the network's only on paper.
        tensor = self._converter.board_to_input_tensor(chess_board)
        policy, _value = self._net.predict(tensor)  # type: ignore[arg-type]
        uci = self._converter.output_tensor_to_move(policy)  # type: ignore[arg-type]

        from_row, from_col = algebraic_to_indices(uci[0:2])
        to_row,   to_col   = algebraic_to_indices(uci[2:4])

        if len(uci) > 4:
            promo = _PROMO_WORD.get(uci[4])
        else:
            # The converter strips the 'q' suffix from queen promotions, so a
            # pawn reaching the last rank with no suffix is a queen promotion.
            piece = board_state.get_piece(from_row, from_col)
            promo = "queen" if (piece and piece[1] == "pawn" and to_row in (0, 7)) else None

        return Move(from_row, from_col, to_row, to_col, promo)

    def _board_with_history(
        self, board_state: BoardState, move_history: list[Move] | None
    ) -> chess.Board:
        """Build a python-chess board carrying a move stack, so the converter can
        fill the network's 7-move history planes (channels 13-103).

        Replays the game from the standard start; python-chess then derives
        castling / en-passant / repetition authoritatively. Falls back to a
        stackless board straight from the FEN if there's no history or the replay
        doesn't land on the current position (e.g. a non-standard start FEN).
        """
        target = chess.Board(board_state.to_fen())
        if not move_history:
            return target
        board = chess.Board()
        try:
            for m in move_history:
                board.push_uci(_custom_move_to_uci(m))
        except ValueError:
            return target            # a move failed to convert/replay
        if board.board_fen() != target.board_fen():
            return target            # desync (e.g. non-standard start) — be safe
        return board

    def name(self) -> str:
        return "BigNetwork (SL)"
