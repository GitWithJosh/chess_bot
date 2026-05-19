import chess


class Board:
    EMPTY = 0
    WHITE = 1
    BLACK = 2

    def __init__(self):
        self.turn = self.WHITE
        self.output_index = {}
        self.board = chess.Board()

    def set_starting_position(self):
        self.board.reset()

    def get_possible_moves(self):
        return list(self.board.legal_moves)

    def apply_move(self, move):
        self.board.push(move)
        self.turn = self.BLACK if self.turn == self.WHITE else self.WHITE

    def is_terminal(self):
        "Return a Tuple (is_terminal: bool, outcome: str) where outcome is 'win', 'loss', 'draw', or None if not terminal."
        if not self.board.is_game_over():
            return False, None

        if self.board.is_checkmate():
            return True, "win" if self.turn == self.WHITE else "loss"
        elif (
            self.board.is_stalemate()
            or self.board.is_insufficient_material()
            or self.board.can_claim_fifty_moves()
            or self.board.can_claim_threefold_repetition()
        ):
            return True, "draw"
        else:
            return True, None

    def to_network_input(self):
        "Convert the current board state to a format suitable for neural network input."
        # This is a placeholder implementation. In a real implementation, you would
        # convert the board state into a tensor or array format that your neural
        # network can process, encoding piece positions, turn, castling rights, etc.
        return str(self.board)


if __name__ == "__main__":
    board = Board()
    board.set_starting_position()
    print(board.board)
