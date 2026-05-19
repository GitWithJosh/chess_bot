import chess
import chess.pgn
import numpy as np
import os
import logging
from typing import List, Tuple, Optional
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Converter:
    """Converts chess board states and moves to neural network input/output formats
    - should take a chess.Board object and convert it into Bit-Planes as Tensor
    - Note: This is unedited Code from Danny
    """

    def __init__(self):
        """Initialize the chess data processor with move mapping"""
        self.move_to_index = {}
        self.index_to_move = {}
        self._build_move_mapping()

    def _build_move_mapping(self):
        """Build mapping from moves to indices (AlphaZero-style)"""
        index = 0

        # Queen moves (56 directions × 7 squares = 392)
        directions = [
            (0, 1),
            (0, -1),
            (1, 0),
            (-1, 0),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]
        for direction in directions:
            for distance in range(1, 8):
                for from_rank in range(8):
                    for from_file in range(8):
                        to_rank = from_rank + direction[0] * distance
                        to_file = from_file + direction[1] * distance
                        if 0 <= to_rank < 8 and 0 <= to_file < 8:
                            from_square = chess.square(from_file, from_rank)
                            to_square = chess.square(to_file, to_rank)
                            move_key = f"{from_square}_{to_square}"
                            if move_key not in self.move_to_index:
                                self.move_to_index[move_key] = index
                                self.index_to_move[index] = move_key
                                index += 1

        # Knight moves (8 directions)
        knight_moves = [
            (2, 1),
            (2, -1),
            (-2, 1),
            (-2, -1),
            (1, 2),
            (1, -2),
            (-1, 2),
            (-1, -2),
        ]
        for knight_move in knight_moves:
            for from_rank in range(8):
                for from_file in range(8):
                    to_rank = from_rank + knight_move[0]
                    to_file = from_file + knight_move[1]
                    if 0 <= to_rank < 8 and 0 <= to_file < 8:
                        from_square = chess.square(from_file, from_rank)
                        to_square = chess.square(to_file, to_rank)
                        move_key = f"{from_square}_{to_square}"
                        if move_key not in self.move_to_index:
                            self.move_to_index[move_key] = index
                            self.index_to_move[index] = move_key
                            index += 1

        # Promotions (simplified - just queen promotions for now)
        for from_file in range(8):
            # White pawn promotions
            from_square = chess.square(from_file, 6)
            to_square = chess.square(from_file, 7)
            move_key = f"{from_square}_{to_square}_q"
            if move_key not in self.move_to_index:
                self.move_to_index[move_key] = index
                self.index_to_move[index] = move_key
                index += 1

            # Black pawn promotions
            from_square = chess.square(from_file, 1)
            to_square = chess.square(from_file, 0)
            move_key = f"{from_square}_{to_square}_q"
            if move_key not in self.move_to_index:
                self.move_to_index[move_key] = index
                self.index_to_move[index] = move_key
                index += 1

        logger.info(f"Built move mapping with {len(self.move_to_index)} unique moves")

    def board_to_tensor(self, board: chess.Board) -> np.ndarray:
        """Convert chess board to 8x8x18 tensor representation

        Args:
            board: Chess board position

        Returns:
            numpy array of shape (8, 8, 18) representing the board state
        """
        tensor = np.zeros((8, 8, 18), dtype=getattr(np, "float16"))

        # Piece locations (channels 0-11)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank, file = divmod(square, 8)
                piece_type = piece.piece_type - 1  # 0-5 for pawn-king
                color_offset = 0 if piece.color == chess.WHITE else 6
                tensor[rank, file, piece_type + color_offset] = 1

        # Castling rights (channels 12-15)
        tensor[:, :, 12] = float(board.has_kingside_castling_rights(chess.WHITE))
        tensor[:, :, 13] = float(board.has_queenside_castling_rights(chess.WHITE))
        tensor[:, :, 14] = float(board.has_kingside_castling_rights(chess.BLACK))
        tensor[:, :, 15] = float(board.has_queenside_castling_rights(chess.BLACK))

        # En passant target square (channel 16)
        if board.ep_square is not None:
            ep_rank, ep_file = divmod(board.ep_square, 8)
            tensor[ep_rank, ep_file, 16] = 1

        # Side to move (channel 17)
        tensor[:, :, 17] = float(board.turn == chess.WHITE)

        return tensor

    def encode_move(self, move: chess.Move) -> int:
        """Encode chess move to index using the move mapping

        Args:
            move: Chess move to encode

        Returns:
            Integer index representing the move (0 if move not found)
        """
        from_square = move.from_square
        to_square = move.to_square

        if move.promotion:
            promotion_piece = chess.piece_symbol(move.promotion)
            move_key = f"{from_square}_{to_square}_{promotion_piece}"
        else:
            move_key = f"{from_square}_{to_square}"

        return self.move_to_index.get(move_key, 0)  # Return 0 if move not found


if __name__ == "__main__":
    converter = Converter()
    board = chess.Board()
    tensor = converter.board_to_tensor(board)
    logger.info(f"Board tensor shape: {tensor.shape}")
    move = chess.Move.from_uci("e2e4")
    move_index = converter.encode_move(move)
    logger.info(f"Encoded move index for e2e4: {move_index}")
