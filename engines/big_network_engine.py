"""BigNetwork engine — plays via one of the trained checkpoints.

Two ways to move, chosen with `mode`:

  "policy"  one forward pass, then the highest-probability legal move. Answers
            in well under a tenth of a second and is clearly the weaker of the
            two, which is the point of offering it.
  "mcts"    a PUCT search over `sims` positions before moving. This is how every
            Elo figure in the report was measured, at 1000 simulations.

Bridges the GUI's own board objects to the python-chess and Converter stack the
network was trained against.
"""

import os
import sys

# big_network.py imports from networks/, monte_carlo_tree_search/ and helpers/,
# all of which live under reinforcement_learning/. Put the repo root and that
# package dir on sys.path before pulling anything in. This is the only place in
# the application that needs it.
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

DEFAULT_SIMS = 200


def _custom_move_to_uci(move: Move) -> str:
    """Custom Move -> UCI string (e.g. e2e4, e7e8q, e1g1 for castling)."""
    uci = move.from_square + move.to_square
    if move.promotion_piece:
        uci += _PROMO_CHAR.get(move.promotion_piece, "")
    return uci


class BigNetworkEngine(ChessEngine):
    """Wraps a trained BigNetwork checkpoint as a playable engine."""

    def __init__(
        self,
        weights_path: str,
        mode: str = "mcts",
        sims: int = DEFAULT_SIMS,
        batch_size: int = 16,
        display_name: str | None = None,
    ):
        if mode not in ("policy", "mcts"):
            raise ValueError(f"mode must be 'policy' or 'mcts', got {mode!r}")

        self._net = BigNetwork()
        self._net.load(weights_path)
        self._converter = Converter()
        self._weights_path = weights_path
        self._mode = mode
        self._sims = max(1, int(sims))
        self._batch_size = batch_size
        self._display_name = display_name or os.path.basename(weights_path).replace(
            ".weights.h5", ""
        )

    def get_best_move(
        self, board_state: BoardState, move_history: list[Move] | None = None
    ) -> Move | None:
        chess_board = self._board_with_history(board_state, move_history)
        if chess_board.is_game_over():
            return None

        if self._mode == "mcts":
            uci = self._net.search_for_best_move(
                chess_board, num_simulations=self._sims, batch_size=self._batch_size
            ).uci()
        else:
            uci = self._policy_move(chess_board)

        return self._uci_to_custom_move(uci, board_state)

    def _policy_move(self, chess_board: chess.Board) -> str:
        """One forward pass, highest-probability legal move."""
        # Converter masks illegal moves against this board and mirrors for Black.
        self._converter.board = chess_board

        # Converter is annotated tf.Tensor but builds/consumes numpy at runtime,
        # so its signatures disagree with the network's only on paper.
        tensor = self._converter.board_to_input_tensor(chess_board)
        policy, _value = self._net.predict(tensor)  # type: ignore[arg-type]
        return self._converter.output_tensor_to_move(policy)  # type: ignore[arg-type]

    def _uci_to_custom_move(self, uci: str, board_state: BoardState) -> Move:
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
        """Build a python-chess board carrying a move stack.

        The 20-plane encoding is position-only, so the stack feeds nothing into
        the network itself. It matters for the search, which needs python-chess
        to recognise threefold repetition and the fifty-move rule, and for the
        game-over test above.

        Replays from the standard start. Falls back to a stackless board built
        from the FEN if there is no history or the replay does not land on the
        current position, e.g. from a non-standard start.
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
        if self._mode == "policy":
            return f"{self._display_name} (policy)"
        return f"{self._display_name} ({self._sims} sims)"
