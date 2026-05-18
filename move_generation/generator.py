"""Legal move generation and validation."""

from board.board import BoardState
from move_generation.move import Move
from utils.coordinates import is_valid_square


class MoveGenerator:
    """Generate legal moves for a position."""

    def __init__(self, board: BoardState):
        self.board = board
        self.color = board.active_color

    def get_legal_moves(self) -> list[Move]:
        """Get all legal moves in current position."""
        pseudo_legal = self.get_pseudo_legal_moves()
        legal = []
        for move in pseudo_legal:
            if self._is_legal(move):
                legal.append(move)
        return legal

    def get_pseudo_legal_moves(self) -> list[Move]:
        """Get all pseudo-legal moves (before checking/stalemate validation)."""
        moves = []
        for row in range(8):
            for col in range(8):
                piece = self.board.get_piece(row, col)
                if piece and piece[0] == self.color:
                    moves.extend(self._moves_from_square(row, col))
        return moves

    def _moves_from_square(self, row: int, col: int) -> list[Move]:
        """Get pseudo-legal moves from a square."""
        piece = self.board.get_piece(row, col)
        if not piece:
            return []

        color, piece_type = piece
        if piece_type == 'pawn':
            return self._pawn_moves(row, col, color)
        elif piece_type == 'knight':
            return self._knight_moves(row, col, color)
        elif piece_type == 'bishop':
            return self._bishop_moves(row, col, color)
        elif piece_type == 'rook':
            return self._rook_moves(row, col, color)
        elif piece_type == 'queen':
            return self._queen_moves(row, col, color)
        elif piece_type == 'king':
            return self._king_moves(row, col, color)
        return []

    def _pawn_moves(self, row: int, col: int, color: str) -> list[Move]:
        """Generate pseudo-legal pawn moves."""
        moves = []
        direction = 1 if color == 'black' else -1
        start_row = 1 if color == 'black' else 6

        # Forward move
        new_row = row + direction
        if is_valid_square(new_row, col) and self.board.get_piece(new_row, col) is None:
            if new_row == 0 or new_row == 7:
                for promo in ['queen', 'rook', 'bishop', 'knight']:
                    moves.append(Move(row, col, new_row, col, promo))
            else:
                moves.append(Move(row, col, new_row, col))

                # Double forward from start
                if row == start_row:
                    double_row = row + 2 * direction
                    if self.board.get_piece(double_row, col) is None:
                        moves.append(Move(row, col, double_row, col))

        # Captures
        for dc in [-1, 1]:
            new_col = col + dc
            new_row = row + direction
            if is_valid_square(new_row, new_col):
                target = self.board.get_piece(new_row, new_col)
                if target and target[0] != color:
                    if new_row == 0 or new_row == 7:
                        for promo in ['queen', 'rook', 'bishop', 'knight']:
                            moves.append(Move(row, col, new_row, new_col, promo))
                    else:
                        moves.append(Move(row, col, new_row, new_col))

        # En passant
        if self.board.en_passant_square:
            from utils.coordinates import algebraic_to_indices
            try:
                ep_row, ep_col = algebraic_to_indices(self.board.en_passant_square)
                if abs(ep_col - col) == 1 and row + direction == ep_row:
                    moves.append(Move(row, col, ep_row, ep_col))
            except (ValueError, IndexError):
                pass

        return moves

    def _knight_moves(self, row: int, col: int, color: str) -> list[Move]:
        """Generate pseudo-legal knight moves."""
        moves = []
        knight_offsets = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1),
        ]
        for dr, dc in knight_offsets:
            new_row, new_col = row + dr, col + dc
            if is_valid_square(new_row, new_col):
                target = self.board.get_piece(new_row, new_col)
                if target is None or target[0] != color:
                    moves.append(Move(row, col, new_row, new_col))
        return moves

    def _bishop_moves(self, row: int, col: int, color: str) -> list[Move]:
        """Generate pseudo-legal bishop moves."""
        return self._sliding_moves(row, col, color, [(-1, -1), (-1, 1), (1, -1), (1, 1)])

    def _rook_moves(self, row: int, col: int, color: str) -> list[Move]:
        """Generate pseudo-legal rook moves."""
        return self._sliding_moves(row, col, color, [(-1, 0), (1, 0), (0, -1), (0, 1)])

    def _queen_moves(self, row: int, col: int, color: str) -> list[Move]:
        """Generate pseudo-legal queen moves."""
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1),
        ]
        return self._sliding_moves(row, col, color, directions)

    def _sliding_moves(
        self, row: int, col: int, color: str, directions: list[tuple[int, int]]
    ) -> list[Move]:
        """Generate sliding piece moves in given directions."""
        moves = []
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            while is_valid_square(new_row, new_col):
                target = self.board.get_piece(new_row, new_col)
                if target is None:
                    moves.append(Move(row, col, new_row, new_col))
                elif target[0] != color:
                    moves.append(Move(row, col, new_row, new_col))
                    break
                else:
                    break
                new_row += dr
                new_col += dc
        return moves

    def _king_moves(self, row: int, col: int, color: str) -> list[Move]:
        """Generate pseudo-legal king moves (including castling)."""
        moves = []

        # Regular moves
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                new_row, new_col = row + dr, col + dc
                if is_valid_square(new_row, new_col):
                    target = self.board.get_piece(new_row, new_col)
                    if target is None or target[0] != color:
                        moves.append(Move(row, col, new_row, new_col))

        # Castling
        if color == 'white':
            castling = self.board.castling_rights
            if 'K' in castling and self._can_castle_kingside(color):
                moves.append(Move(row, col, row, 6))
            if 'Q' in castling and self._can_castle_queenside(color):
                moves.append(Move(row, col, row, 2))
        else:
            castling = self.board.castling_rights
            if 'k' in castling and self._can_castle_kingside(color):
                moves.append(Move(row, col, row, 6))
            if 'q' in castling and self._can_castle_queenside(color):
                moves.append(Move(row, col, row, 2))

        return moves

    def _can_castle_kingside(self, color: str) -> bool:
        """Check if kingside castling is possible."""
        row = 0 if color == 'black' else 7
        king = self.board.get_piece(row, 4)
        rook = self.board.get_piece(row, 7)
        if king != (color, 'king') or rook != (color, 'rook'):
            return False

        if self.board.get_piece(row, 5) is not None or self.board.get_piece(row, 6) is not None:
            return False

        if self._is_king_attacked(self.board, color):
            return False

        through_board = self.board.copy()
        through_board.set_piece(row, 4, None)
        through_board.set_piece(row, 5, (color, 'king'))
        if self._is_king_attacked(through_board, color):
            return False

        destination_board = self.board.copy()
        self._apply_move_to_board(destination_board, Move(row, 4, row, 6))
        return not self._is_king_attacked(destination_board, color)

    def _can_castle_queenside(self, color: str) -> bool:
        """Check if queenside castling is possible."""
        row = 0 if color == 'black' else 7
        king = self.board.get_piece(row, 4)
        rook = self.board.get_piece(row, 0)
        if king != (color, 'king') or rook != (color, 'rook'):
            return False

        if (
            self.board.get_piece(row, 1) is not None
            or self.board.get_piece(row, 2) is not None
            or self.board.get_piece(row, 3) is not None
        ):
            return False

        if self._is_king_attacked(self.board, color):
            return False

        through_board = self.board.copy()
        through_board.set_piece(row, 4, None)
        through_board.set_piece(row, 3, (color, 'king'))
        if self._is_king_attacked(through_board, color):
            return False

        destination_board = self.board.copy()
        self._apply_move_to_board(destination_board, Move(row, 4, row, 2))
        return not self._is_king_attacked(destination_board, color)

    def _is_legal(self, move: Move) -> bool:
        """Check if move is legal (doesn't leave king in check)."""
        test_board = self.board.copy()
        self._apply_move_to_board(test_board, move)
        return not self._is_king_attacked(test_board, self.color)

    def _apply_move_to_board(self, board: BoardState, move: Move):
        """Apply move to board state (mutates)."""
        piece = board.get_piece(move.from_row, move.from_col)
        target = board.get_piece(move.to_row, move.to_col)
        board.set_piece(move.from_row, move.from_col, None)
        board.set_piece(move.to_row, move.to_col, piece)

        # Castling: move rook as well
        if piece and piece[1] == 'king' and abs(move.to_col - move.from_col) == 2:
            rook_from_col = 7 if move.to_col > move.from_col else 0
            rook_to_col = 5 if move.to_col > move.from_col else 3
            rook = board.get_piece(move.to_row, rook_from_col)
            board.set_piece(move.to_row, rook_from_col, None)
            board.set_piece(move.to_row, rook_to_col, rook)

        # En passant: remove the captured pawn (it sits on the same row as the
        # moving pawn, not on the destination square)
        if piece and piece[1] == 'pawn' and target is None and move.from_col != move.to_col:
            board.set_piece(move.from_row, move.to_col, None)

        # Promotion: replace pawn with promoted piece
        if piece and piece[1] == 'pawn' and move.promotion_piece:
            board.set_piece(move.to_row, move.to_col, (piece[0], move.promotion_piece))

    def _is_king_attacked(self, board: BoardState, color: str) -> bool:
        """Check if color's king is under attack (doesn't use recursion)."""
        king_pos = None
        for row in range(8):
            for col in range(8):
                piece = board.get_piece(row, col)
                if piece and piece[0] == color and piece[1] == 'king':
                    king_pos = (row, col)
                    break
            if king_pos:
                break

        if not king_pos:
            return False

        opponent = 'black' if color == 'white' else 'white'

        for row in range(8):
            for col in range(8):
                piece = board.get_piece(row, col)
                if piece and piece[0] == opponent:
                    if self._piece_attacks(board, row, col, king_pos[0], king_pos[1], opponent):
                        return True

        return False

    def _piece_attacks(
        self, board: BoardState, from_row: int, from_col: int, to_row: int, to_col: int, color: str
    ) -> bool:
        """Check if piece at (from_row, from_col) can attack (to_row, to_col)."""
        piece = board.get_piece(from_row, from_col)
        if not piece or piece[0] != color:
            return False

        piece_type = piece[1]

        if piece_type == 'pawn':
            direction = 1 if color == 'black' else -1
            if from_row + direction == to_row and abs(from_col - to_col) == 1:
                return True

        elif piece_type == 'knight':
            dr = abs(from_row - to_row)
            dc = abs(from_col - to_col)
            if (dr == 2 and dc == 1) or (dr == 1 and dc == 2):
                return True

        elif piece_type == 'king':
            if abs(from_row - to_row) <= 1 and abs(from_col - to_col) <= 1:
                return True

        elif piece_type == 'bishop':
            if self._is_diagonal_clear(board, from_row, from_col, to_row, to_col):
                return True

        elif piece_type == 'rook':
            if self._is_straight_clear(board, from_row, from_col, to_row, to_col):
                return True

        elif piece_type == 'queen':
            if (self._is_diagonal_clear(board, from_row, from_col, to_row, to_col) or
                self._is_straight_clear(board, from_row, from_col, to_row, to_col)):
                return True

        return False

    def _is_diagonal_clear(self, board: BoardState, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
        """Check if diagonal path is clear."""
        if abs(from_row - to_row) != abs(from_col - to_col):
            return False

        dr = 0 if to_row == from_row else (1 if to_row > from_row else -1)
        dc = 0 if to_col == from_col else (1 if to_col > from_col else -1)

        r, c = from_row + dr, from_col + dc
        while (r, c) != (to_row, to_col):
            if board.get_piece(r, c) is not None:
                return False
            r += dr
            c += dc

        return True

    def _is_straight_clear(self, board: BoardState, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
        """Check if straight path is clear."""
        if from_row != to_row and from_col != to_col:
            return False

        if from_row == to_row:
            step = 1 if to_col > from_col else -1
            for c in range(from_col + step, to_col, step):
                if board.get_piece(from_row, c) is not None:
                    return False
        else:
            step = 1 if to_row > from_row else -1
            for r in range(from_row + step, to_row, step):
                if board.get_piece(r, from_col) is not None:
                    return False

        return True
