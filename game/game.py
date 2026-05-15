"""Game state and rules enforcement."""

from enum import Enum
from board.board import BoardState
from move_generation.move import Move
from move_generation.generator import MoveGenerator


class GameStatus(Enum):
    ONGOING = "ongoing"
    CHECKMATE = "checkmate"
    STALEMATE = "stalemate"
    THREEFOLD = "threefold_repetition"
    FIFTY_MOVE = "fifty_move_rule"
    INSUFFICIENT = "insufficient_material"


class Game:
    """Manages game state, rules, and move history."""

    def __init__(self, starting_fen: str | None = None):
        if starting_fen:
            self.board = BoardState.from_fen(starting_fen)
        else:
            self.board = BoardState.starting_position()
        self.move_history: list[Move] = []
        self.position_history: list[str] = [self.board.to_fen()]

    def get_legal_moves(self) -> list[Move]:
        """Get all legal moves for current position."""
        gen = MoveGenerator(self.board)
        return gen.get_legal_moves()

    def make_move(self, move: Move) -> bool:
        """Apply move to game state. Returns True if valid, False otherwise."""
        legal_moves = self.get_legal_moves()
        if move not in legal_moves:
            return False

        old_board = self.board.copy()
        self._apply_move(move)
        self.move_history.append(move)
        self.position_history.append(self.board.to_fen())

        return True

    def _apply_move(self, move: Move):
        """Apply move to board and update game state."""
        piece = self.board.get_piece(move.from_row, move.from_col)
        target = self.board.get_piece(move.to_row, move.to_col)

        # Handle castling
        if piece[1] == "king" and abs(move.to_col - move.from_col) == 2:
            self._handle_castling(move)
        else:
            self.board.set_piece(move.from_row, move.from_col, None)
            self.board.set_piece(move.to_row, move.to_col, piece)

        # Handle pawn promotion
        if piece[1] == "pawn" and move.promotion_piece:
            self.board.set_piece(
                move.to_row, move.to_col, (piece[0], move.promotion_piece)
            )

        # Handle en passant
        if piece[1] == "pawn" and target is None and move.from_col != move.to_col:
            ep_capture_row = move.from_row
            self.board.set_piece(ep_capture_row, move.to_col, None)

        self._update_game_state(move, piece, target)

    def _handle_castling(self, move: Move):
        """Move king and rook for castling."""
        king_piece = self.board.get_piece(move.from_row, move.from_col)
        self.board.set_piece(move.from_row, move.from_col, None)
        self.board.set_piece(move.to_row, move.to_col, king_piece)

        if move.to_col == 6:
            rook = self.board.get_piece(move.from_row, 7)
            self.board.set_piece(move.from_row, 7, None)
            self.board.set_piece(move.from_row, 5, rook)
        else:
            rook = self.board.get_piece(move.from_row, 0)
            self.board.set_piece(move.from_row, 0, None)
            self.board.set_piece(move.from_row, 3, rook)

    def _update_game_state(self, move: Move, piece: tuple[str, str], target):
        """Update castling rights, en passant, halfmove clock, turn."""
        color = piece[0]
        new_rights = self.board.castling_rights

        if piece[1] == "king":
            if color == "white":
                new_rights = new_rights.replace("K", "").replace("Q", "")
            else:
                new_rights = new_rights.replace("k", "").replace("q", "")
        elif piece[1] == "rook":
            if color == "white":
                if move.from_col == 0:
                    new_rights = new_rights.replace("Q", "")
                elif move.from_col == 7:
                    new_rights = new_rights.replace("K", "")
            else:
                if move.from_col == 0:
                    new_rights = new_rights.replace("q", "")
                elif move.from_col == 7:
                    new_rights = new_rights.replace("k", "")

        self.board.castling_rights = new_rights if new_rights else "-"

        self.board.en_passant_square = None
        if piece[1] == "pawn" and abs(move.to_row - move.from_row) == 2:
            from utils.coordinates import indices_to_algebraic

            ep_row = (move.from_row + move.to_row) // 2
            ep_col = move.to_col
            self.board.en_passant_square = indices_to_algebraic(ep_row, ep_col)

        if piece[1] == "pawn" or target:
            self.board.halfmove_clock = 0
        else:
            self.board.halfmove_clock += 1

        self.board.active_color = "black" if color == "white" else "white"
        if color == "black":
            self.board.fullmove_number += 1

    def get_status(self) -> GameStatus:
        """Get current game status (ongoing, checkmate, stalemate, etc.)."""
        legal_moves = self.get_legal_moves()
        in_check = self._is_in_check(self.board.active_color)

        if len(legal_moves) == 0:
            return GameStatus.CHECKMATE if in_check else GameStatus.STALEMATE

        if self.board.halfmove_clock >= 100:
            return GameStatus.FIFTY_MOVE

        if self._check_threefold_repetition():
            return GameStatus.THREEFOLD

        if self._is_insufficient_material():
            return GameStatus.INSUFFICIENT

        return GameStatus.ONGOING

    def _is_in_check(self, color: str) -> bool:
        """Check if color is under attack."""
        gen = MoveGenerator(self.board)
        return gen._is_king_attacked(self.board, color)

    def _check_threefold_repetition(self) -> bool:
        """Check if position repeated 3 times."""
        # Threefold repetition compares piece placement, side to move,
        # castling rights and en-passant square — ignore halfmove/fullmove clocks.
        if len(self.position_history) < 3:
            return False

        def _repetition_key(fen: str) -> str:
            parts = fen.split()
            # FEN format: placement, active, castling, en-passant, halfmove, fullmove
            return " ".join(parts[:4])

        current_key = _repetition_key(self.position_history[-1])
        keys = (_repetition_key(f) for f in self.position_history)
        count = sum(1 for k in keys if k == current_key)

        return count >= 3

    def _is_insufficient_material(self) -> bool:
        """Check for insufficient mating material."""
        white_pieces = []
        black_pieces = []
        for row in range(8):
            for col in range(8):
                piece = self.board.get_piece(row, col)
                if piece:
                    if piece[0] == "white":
                        white_pieces.append(piece[1])
                    else:
                        black_pieces.append(piece[1])

        white_non_king = [p for p in white_pieces if p != "king"]
        black_non_king = [p for p in black_pieces if p != "king"]

        if not white_non_king or not black_non_king:
            return len(white_non_king) == 0 and len(black_non_king) == 0

        if len(white_non_king) == 1 and len(black_non_king) == 1:
            return white_non_king[0] in ["bishop", "knight"] and black_non_king[0] in [
                "bishop",
                "knight",
            ]

        return False

    def undo_move(self) -> bool:
        """Undo last move if possible."""
        if not self.move_history:
            return False

        self.move_history.pop()
        self.position_history.pop()
        if self.position_history:
            self.board = BoardState.from_fen(self.position_history[-1])
        return True

    def __str__(self) -> str:
        """Pretty print board."""
        lines = []
        for row in range(8):
            line = ""
            for col in range(8):
                piece = self.board.get_piece(row, col)
                if piece is None:
                    line += ". "
                else:
                    color, piece_type = piece
                    char = piece_type[0].upper() if color == "white" else piece_type[0]
                    line += char + " "
            lines.append(line)
        return "\n".join(lines)
