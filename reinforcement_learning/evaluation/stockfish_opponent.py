import chess
from stockfish import Stockfish

from engine_opponent import EngineOpponent

# Requires: "sudo apt install stockfish" to get binary and "pip install stockfish"/"uv add stockfish"


class StockfishOpponent(EngineOpponent):

    PRESETS = {
        "beginner": {"skill_level": 0, "depth": 1},
        "easy": {"skill_level": 5, "depth": 5},
        "intermediate": {"skill_level": 10, "depth": 10},
        "hard": {"skill_level": 15, "depth": 15},
        "maximum": {"skill_level": 20, "depth": 20},
    }

    def __init__(
        self,
        board: chess.Board,
        stockfish_path: str = None,
        level: str = "beginner",
        depth: int = None,
        think_time: int = None,
    ):
        self.board = board
        self.think_time = think_time

        preset = self.PRESETS.get(level, self.PRESETS["beginner"])
        self.depth = depth if depth is not None else preset["depth"]

        params = {"Skill Level": preset["skill_level"]}

        if stockfish_path:
            self.engine = Stockfish(path=stockfish_path, parameters=params)
        else:
            self.engine = Stockfish(parameters=params)

    def choose_move(self, board) -> chess.Move:
        self.board = board
        self.engine.set_fen_position(self.board.fen())

        if self.think_time:
            best_move_uci = self.engine.get_best_move_time(self.think_time)
        else:
            best_move_uci = self.engine.get_best_move(self.depth)

        return chess.Move.from_uci(best_move_uci)
