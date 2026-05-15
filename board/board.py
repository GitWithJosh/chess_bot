"""Chess board state representation and manipulation."""

from copy import deepcopy
from dataclasses import dataclass
from utils.fen import parse_full_fen, board_to_fen
from utils.coordinates import is_valid_square


@dataclass
class BoardState:
    """Immutable chess board state for move generation and ML compatibility."""

    board: list[list]
    active_color: str
    castling_rights: str
    en_passant_square: str | None
    halfmove_clock: int
    fullmove_number: int

    def copy(self) -> 'BoardState':
        """Create a deep copy of board state."""
        return BoardState(
            board=deepcopy(self.board),
            active_color=self.active_color,
            castling_rights=self.castling_rights,
            en_passant_square=self.en_passant_square,
            halfmove_clock=self.halfmove_clock,
            fullmove_number=self.fullmove_number,
        )

    def get_piece(self, row: int, col: int) -> tuple[str, str] | None:
        """Get piece at square or None."""
        if not is_valid_square(row, col):
            return None
        return self.board[row][col]

    def set_piece(self, row: int, col: int, piece: tuple[str, str] | None):
        """Set piece at square (mutates copy)."""
        if is_valid_square(row, col):
            self.board[row][col] = piece

    def to_fen(self) -> str:
        """Export board to FEN string."""
        board_fen = board_to_fen(self.board)
        active = 'w' if self.active_color == 'white' else 'b'
        return (
            f"{board_fen} {active} {self.castling_rights} "
            f"{self.en_passant_square or '-'} {self.halfmove_clock} {self.fullmove_number}"
        )

    @staticmethod
    def from_fen(fen: str) -> 'BoardState':
        """Create board state from FEN string."""
        parts = parse_full_fen(fen)
        active_color = 'white' if parts['active_color'] == 'w' else 'black'
        return BoardState(
            board=parts['board'],
            active_color=active_color,
            castling_rights=parts['castling'],
            en_passant_square=parts['en_passant'] if parts['en_passant'] != '-' else None,
            halfmove_clock=parts['halfmove_clock'],
            fullmove_number=parts['fullmove_number'],
        )

    @staticmethod
    def starting_position() -> 'BoardState':
        """Return starting chess position."""
        return BoardState.from_fen(
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
        )
