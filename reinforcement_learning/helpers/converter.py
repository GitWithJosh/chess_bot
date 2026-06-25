import chess
import numpy as np
import json
import tensorflow as tf

class Converter:
    """
    Class for converting stuff
    """
    def __init__(self, lookup_path: str="reinforcement_learning/move_lookup.json"):
        self.board = chess.Board()
        self.lookup = {}
        with open(lookup_path, "r") as f:
            self.lookup = json.load(f)

        self.index_lookup = {v: int(k) for k, v in self.lookup.items()}


    def board_to_input_tensor(self, board: chess.Board) -> np.ndarray:
        """Convert a chess board to an (8, 8, 20) position-only tensor.

            Position-only encoding — no move history. Every plane is derivable
            from the current board alone (i.e. from a single FEN).

            0-5  : friendly pieces (pawn, knight, bishop, rook, queen, king)
            6-11 : enemy pieces    (pawn, knight, bishop, rook, queen, king)
            12   : friendly queenside castling right (all-ones if present)
            13   : friendly kingside  castling right
            14   : enemy    queenside castling right
            15   : enemy    kingside  castling right
            16   : side to move (1 if it is black's turn in the real game)
            17   : 50-move rule — halfmove clock / 100, broadcast across the plane
            18   : en passant target square — single 1 on the ep square (only if a
                   legal en passant capture exists)
            19   : all-ones (helps the convolutions detect board edges)

            The board is oriented so the side to move is always "white"/friendly
            (mirrored when black is to move); mirror() also flips the ep square and
            swaps piece colours, so friendly pieces are always white in this frame.

            All planes have dimension 8x8.

            Args:
                board: Chess board position

            Returns:
                numpy array of shape (8, 8, 20) representing the board state
        """
        if board.turn == chess.BLACK:
            oriented_board = board.mirror()
        else:
            oriented_board = board.copy()

        tensor = np.zeros((8, 8, 20), dtype=np.float16)

        # After orientation the friendly (side-to-move) pieces are always white.
        friendly_color = chess.WHITE

        # Channels 0-11: piece placement for the current position.
        # Iterate only the ~32 occupied squares, not all 64 with piece_at().
        for square, piece in oriented_board.piece_map().items():
            rank = square // 8
            file = square % 8
            ptype = piece.piece_type - 1  # 0-5
            if piece.color == friendly_color:
                # Channels 0-5: friendly pieces
                tensor[rank, file, ptype] = 1
            else:
                # Channels 6-11: enemy pieces
                tensor[rank, file, 6 + ptype] = 1

        # Channels 12-15: castling rights (oriented frame, so white == friendly)
        tensor[:, :, 12] = float(oriented_board.has_queenside_castling_rights(chess.WHITE))
        tensor[:, :, 13] = float(oriented_board.has_kingside_castling_rights(chess.WHITE))
        tensor[:, :, 14] = float(oriented_board.has_queenside_castling_rights(chess.BLACK))
        tensor[:, :, 15] = float(oriented_board.has_kingside_castling_rights(chess.BLACK))

        # Channel 16: side to move (1 if black's turn)
        tensor[:, :, 16] = float(board.turn == chess.BLACK)

        # Channel 17: 50-move rule (halfmove clock / 100, same value everywhere)
        tensor[:, :, 17] = board.halfmove_clock / 100

        # Channel 18: en passant target square (only when a legal ep capture exists)
        if oriented_board.has_legal_en_passant():
            ep = oriented_board.ep_square
            tensor[ep // 8, ep % 8, 18] = 1

        # Channel 19: set to 1 (helps detect board edges)
        tensor[:, :, 19] = 1

        return tensor

    def output_tensor_to_move(self, network_output: tf.Tensor) -> str:
        """
        Takes a network output (raw policy logits) and converts it to a move in
        UCI chess notation.
        """
        move_logits = network_output[:1858]
        legal_move_probabilities = self.mask_illegal_moves(self.board, move_logits)
        best_move_index = tf.argmax(legal_move_probabilities).numpy()

        move_uci = self._get_move_from_index(best_move_index)

        if self.board.turn == chess.BLACK:
            move_uci = self._mirror_move_uci(move_uci)

        return move_uci

    def mask_illegal_moves(self, board: chess.Board, move_logits: tf.Tensor) -> tf.Tensor:
        """
        Mask illegal moves and return a probability distribution over the legal
        moves only (sums to 1).

        The policy head outputs RAW logits, so we mask the logits directly
        (illegal → -inf) and apply a single softmax. No log() step is needed.
        """
        index_lookup = self.index_lookup

        mask = np.zeros(len(move_logits), dtype=bool)

        for move in board.legal_moves:
            move_uci = move.uci()

            # Mirror the move to friendly perspective if black's turn
            if board.turn == chess.BLACK:
                move_uci = self._mirror_move_uci(move_uci)

            # Handle queen promotions (default, not in lookup)
            if move.promotion == chess.QUEEN:
                move_uci = move_uci[:-1]

            if move_uci in index_lookup:
                mask[index_lookup[move_uci]] = True

        # Illegal moves → -inf logit, then a single softmax over the legal logits.
        masked_logits = tf.where(mask, move_logits, tf.constant(float('-inf')))
        return tf.nn.softmax(masked_logits)

    def _mirror_move_uci(self, move_uci:str) -> str:
        def flip_rank(c):
            return str(9-int(c))

        result = move_uci[0] + flip_rank(move_uci[1]) + move_uci[2] + flip_rank(move_uci[3])
        if len(move_uci) > 4:
            result += move_uci[4]  # Promotion piece
        return result


    def _get_move_from_index(self, index) -> str:
        """Returns the corresponding move notation given an index"""
        return self.lookup[str(index)]