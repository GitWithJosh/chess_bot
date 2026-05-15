"""Move representation."""

from dataclasses import dataclass
from utils.coordinates import indices_to_algebraic


@dataclass
class Move:
    """Represents a chess move."""

    from_row: int
    from_col: int
    to_row: int
    to_col: int
    promotion_piece: str | None = None

    @property
    def from_square(self) -> str:
        """Get source square in algebraic notation."""
        return indices_to_algebraic(self.from_row, self.from_col)

    @property
    def to_square(self) -> str:
        """Get target square in algebraic notation."""
        return indices_to_algebraic(self.to_row, self.to_col)

    def __str__(self) -> str:
        """Return move in standard notation (e.g., 'e2e4')."""
        move = f"{self.from_square}{self.to_square}"
        if self.promotion_piece:
            move += self.promotion_piece
        return move

    def __eq__(self, other) -> bool:
        """Check move equality."""
        if not isinstance(other, Move):
            return False
        return (
            self.from_row == other.from_row
            and self.from_col == other.from_col
            and self.to_row == other.to_row
            and self.to_col == other.to_col
            and self.promotion_piece == other.promotion_piece
        )

    def __hash__(self) -> int:
        """Hash move for use in sets/dicts."""
        return hash(
            (self.from_row, self.from_col, self.to_row, self.to_col, self.promotion_piece)
        )
