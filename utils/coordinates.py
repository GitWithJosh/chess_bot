"""Coordinate system helpers for chess board (0-7, algebraic notation)."""


def algebraic_to_indices(notation: str) -> tuple[int, int]:
    """Convert algebraic notation (e.g., 'a1') to (row, col) indices."""
    if len(notation) != 2:
        raise ValueError(f"Invalid notation: {notation}")
    col = ord(notation[0]) - ord('a')
    row = 8 - int(notation[1])
    if not (0 <= row < 8 and 0 <= col < 8):
        raise ValueError(f"Out of bounds: {notation}")
    return row, col


def indices_to_algebraic(row: int, col: int) -> str:
    """Convert (row, col) indices to algebraic notation."""
    if not (0 <= row < 8 and 0 <= col < 8):
        raise ValueError(f"Out of bounds: ({row}, {col})")
    return chr(ord('a') + col) + str(8 - row)


def is_valid_square(row: int, col: int) -> bool:
    """Check if coordinates are within board bounds."""
    return 0 <= row < 8 and 0 <= col < 8
