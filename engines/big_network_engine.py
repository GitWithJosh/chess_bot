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
from monte_carlo_tree_search.nodes_and_edges_v2 import Node


_PROMO_WORD = {"q": "queen", "r": "rook", "b": "bishop", "n": "knight"}

DEFAULT_SIMS = 200


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
        # The top moves of the most recent move, for the search readout.
        self.last_lines: list[dict] = []
        # A forward pass costs about 60 ms and the GUI redraws 30 times a second,
        # so the value head is evaluated once per position and remembered.
        self._wdl_cache: dict[str, tuple[float, float, float]] = {}

    def get_best_move(
        self, board_state: BoardState, move_history: list[Move] | None = None
    ) -> Move | None:
        chess_board = self._board_with_history(board_state, move_history)
        # Deliberately without claim_draw. The caller owns the draw rules, and
        # Game.get_status() only calls threefold once the position has actually
        # occurred three times, while claim_draw already fires a ply earlier
        # when some legal move would produce the third occurrence. Refusing to
        # move there would strand the game one ply short of its own end.
        if chess_board.is_game_over():
            return None

        if self._mode == "mcts":
            move = self._net.search_for_best_move(
                chess_board, num_simulations=self._sims, batch_size=self._batch_size
            )
            node = self._net._search_root
        else:
            move, node = self._policy_move(chess_board)

        if move is None:
            self.last_lines = []
            return None

        self.last_lines = self._read_lines(node, move)
        return self._uci_to_custom_move(move.uci(), board_state)

    def _policy_move(self, chess_board: chess.Board) -> tuple[chess.Move | None, Node]:
        """One forward pass, highest-prior legal move, no search.

        Goes through a Node so the readout has the same edges the search path
        produces. It also hands back a real legal move, which sidesteps the
        converter dropping the suffix from queen promotions.
        """
        node = Node(chess_board.copy())
        node.expand(self._net, self._converter)
        if not node.edges:
            # Node.is_terminal also counts a real threefold repetition and a
            # claimable fifty-move draw, which board.is_game_over() above does
            # not, so expand can decline to create any edges.
            return None, node
        return max(node.edges, key=lambda e: e.P).move, node

    def _read_lines(self, node, chosen: chess.Move) -> list[dict]:
        """The five most-visited moves, as the notebook's table shows them."""
        if node is None or not getattr(node, "edges", None):
            return []
        total = sum(e.N for e in node.edges) or 1
        top = sorted(node.edges, key=lambda e: (e.N, e.P), reverse=True)[:5]
        lines = []
        for e in top:
            try:
                san = node.board.san(e.move)
            except (ValueError, AssertionError):
                san = e.move.uci()
            lines.append({
                "san": san,
                "n": e.N,
                "share": e.N / total,
                "p": e.P,
                "q": e.Q,
                "chosen": e.move == chosen,
            })
        return lines

    def evaluate_wdl(
        self, board_state: BoardState, move_history: list[Move] | None = None
    ) -> tuple[float, float, float]:
        """Win, draw and loss from the point of view of the side to move."""
        board = self._board_with_history(board_state, move_history)
        key = board.fen()
        hit = self._wdl_cache.get(key)
        if hit is not None:
            return hit
        _policy, wdl = self._net.predict(
            self._converter.board_to_input_tensor(board)
        )  # type: ignore[arg-type]
        result = (float(wdl[0]), float(wdl[1]), float(wdl[2]))
        if len(self._wdl_cache) > 512:
            self._wdl_cache.clear()
        self._wdl_cache[key] = result
        return result

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
                board.push_uci(m.to_uci())
        except ValueError:
            return target            # a move failed to convert/replay
        if board.board_fen() != target.board_fen():
            return target            # desync (e.g. non-standard start) — be safe
        return board

    @property
    def mode(self) -> str:
        return self._mode

    def name(self) -> str:
        if self._mode == "policy":
            return f"{self._display_name} (policy)"
        return f"{self._display_name} ({self._sims} sims)"
