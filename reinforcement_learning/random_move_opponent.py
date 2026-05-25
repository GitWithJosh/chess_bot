import chess
from random import choice

class RandomMover:
    def __init__(self, board:chess.Board):
        self.board = board

    def choose_random_move(self) -> chess.Move:
        legal_moves = list(self.board.legal_moves)
        print(legal_moves)
        return choice(legal_moves)