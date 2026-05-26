import chess
from random import choice
from engine_opponent import EngineOpponent


class RandomMover(EngineOpponent):
    def __init__(self):
        self.board = None

    def choose_move(self, board:chess.Board) -> chess.Move:
        self.board = board
        legal_moves = list(self.board.legal_moves)
        print(legal_moves)
        return choice(legal_moves)