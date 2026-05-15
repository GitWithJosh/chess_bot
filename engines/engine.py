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
    def get_best_move(self, board_state: BoardState) -> Move | None:
        """Return best move for current position.

        Args:
            board_state: Current board state

        Returns:
            Move object, or None if no legal moves
        """
        pass

    def name(self) -> str:
        """Engine name for display."""
        return self.__class__.__name__
