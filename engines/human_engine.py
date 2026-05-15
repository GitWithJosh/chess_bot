"""Human input engine (GUI-driven)."""

from board.board import BoardState
from move_generation.move import Move
from engines.engine import ChessEngine


class HumanInputEngine(ChessEngine):
    """Engine that waits for human input from GUI."""

    def __init__(self):
        self.pending_move: Move | None = None

    def get_best_move(self, board_state: BoardState) -> Move | None:
        """Return move set via set_move(), or None if not set."""
        return self.pending_move

    def set_move(self, move: Move):
        """Set the move to be played (called by GUI)."""
        self.pending_move = move

    def clear_move(self):
        """Clear pending move."""
        self.pending_move = None

    def name(self) -> str:
        return "Human"
