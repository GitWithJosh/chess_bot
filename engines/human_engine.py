"""Human input engine (GUI-driven)."""

from board.board import BoardState
from move_generation.move import Move
from engines.engine import ChessEngine


class HumanInputEngine(ChessEngine):
    """Engine that represents a human player controlled via GUI."""

    def get_best_move(self, board_state: BoardState) -> Move | None:
        """
        Human moves are handled directly by the GUI/game layer,
        so this engine does not generate or store moves.
        """
        return None

    def name(self) -> str:
        return "Human"