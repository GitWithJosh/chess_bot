"""Engine interface and implementations."""

from abc import ABC, abstractmethod
from board.board import BoardState
from move_generation.move import Move


class ChessEngine(ABC):
    """Base interface for all chess engines.

    Engines must be:
    - Stateless (no side effects)
    - Deterministic (same position -> same move, unless randomness intended)
    - Independent of GUI
    - Input: board state only
    - Output: legal move
    """

    @abstractmethod
    def get_best_move(
        self, board_state: BoardState, move_history: list[Move] | None = None
    ) -> Move | None:
        """Return best move for current position.

        Args:
            board_state: Current board state
            move_history: Moves played to reach this position, oldest first.
                Optional context for engines that need game history (e.g. a
                neural net that encodes prior positions); position-only engines
                may ignore it.

        Returns:
            Move object, or None if no legal moves
        """
        pass

    def name(self) -> str:
        """Engine name for display."""
        return self.__class__.__name__
