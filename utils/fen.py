"""FEN (Forsyth-Edwards Notation) parsing and generation."""

PIECE_CHARS = {
    'K': ('white', 'king'), 'Q': ('white', 'queen'), 'R': ('white', 'rook'),
    'B': ('white', 'bishop'), 'N': ('white', 'knight'), 'P': ('white', 'pawn'),
    'k': ('black', 'king'), 'q': ('black', 'queen'), 'r': ('black', 'rook'),
    'b': ('black', 'bishop'), 'n': ('black', 'knight'), 'p': ('black', 'pawn'),
}


def fen_to_board(fen_board: str) -> list[list]:
    """Parse FEN board string into 8x8 board representation.

    Returns list of lists where None = empty, (color, piece_type) = piece.
    """
    board = [[None for _ in range(8)] for _ in range(8)]
    rows = fen_board.split('/')

    for row_idx, row in enumerate(rows):
        col_idx = 0
        for char in row:
            if char.isdigit():
                col_idx += int(char)
            elif char in PIECE_CHARS:
                color, piece_type = PIECE_CHARS[char]
                board[row_idx][col_idx] = (color, piece_type)
                col_idx += 1

    return board


def board_to_fen(board: list[list]) -> str:
    """Convert board representation to FEN string."""
    reverse_pieces = {v: k for k, v in PIECE_CHARS.items()}
    fen_rows = []

    for row in board:
        fen_row = ''
        empty = 0
        for piece in row:
            if piece is None:
                empty += 1
            else:
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                fen_row += reverse_pieces[piece]
        if empty > 0:
            fen_row += str(empty)
        fen_rows.append(fen_row)

    return '/'.join(fen_rows)


def parse_full_fen(fen: str) -> dict:
    """Parse complete FEN string into components.

    Returns dict with keys: board, active_color, castling, en_passant, halfmove, fullmove
    """
    parts = fen.split()
    if len(parts) != 6:
        raise ValueError(f"Invalid FEN: expected 6 parts, got {len(parts)}")

    return {
        'board': fen_to_board(parts[0]),
        'active_color': parts[1],
        'castling': parts[2],
        'en_passant': parts[3],
        'halfmove_clock': int(parts[4]),
        'fullmove_number': int(parts[5]),
    }
