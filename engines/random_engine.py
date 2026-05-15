"""Random move engine (baseline)."""

import random
from board.board import BoardState
from move_generation.move import Move
from move_generation.generator import MoveGenerator
from engines.engine import ChessEngine


class RandomEngine(ChessEngine):
    """Engine that plays random legal moves."""

    def get_best_move(self, board_state: BoardState) -> Move | None:
        """Return a random legal move."""
        gen = MoveGenerator(board_state)
        legal_moves = gen.get_legal_moves()
        if not legal_moves:
            return None
        return random.choice(legal_moves)

    def name(self) -> str:
        return "Random"
