import chess
import numpy as np
import json
import tensorflow as tf

class Converter:
    """
    Class for converting stuff
    """
    def __init__(self):
        self.board = chess.Board()
        self.lookup = {}
        with open("reinforcement_learning/move_lookup.json", "r") as f:
            self.lookup = json.load(f)
        # Reverse map (uci -> index), built once. Callers reuse this instead of
        # rebuilding the 1858-entry dict on every node expansion / move mask.
        self.index_lookup = {v: int(k) for k, v in self.lookup.items()}


    def board_to_input_tensor(self, board:chess.Board) -> tf.Tensor:
        """Convert chess board to 8x8x112 tensor representation

            0-11: piece placement both colors (Friendly (pawns, knights, bishops, rooks, queens, king); Enemy (pawns, knights, bishops, rooks, queens, king))
            12: shows if repetition has occurred once or twice (is 1 regardless)
            13-103: Repeat for seven previous positions
            104-107: castling rights (white queenside, white kingside, black queenside, black kingside); all ones if right exists
            108: 1 if blacks turn, 0 whites turn
            109: 50 move rule; number of moves where no capture was made and no pawn was moved; same scalar value in all entries ;Half-move clock divided by 100
            110: Set to 0
            111: Set to 1 (to help detect bord edges)

            All planes have dimension 8x8

            Args:
                board: Chess board position

            Returns:
                numpy array of shape (8, 8, 112) representing the board state
        """
        if board.turn == chess.BLACK:
            oriented_board = board.mirror()
        else:
            oriented_board = board.copy()

        tensor = np.zeros((8, 8, 112), dtype=getattr(np, "float16"))

        # Build a list of boards by walking back through the move stack
        boards = [oriented_board.copy()]
        temp_board = board.copy()
        for _ in range(7):
            if temp_board.move_stack:
                temp_board.pop()
                if board.turn == chess.BLACK:
                    boards.append(temp_board.mirror())
                else:
                    boards.append(temp_board.copy())
            else:
                boards.append(None)  # No history available

        # Channels 0-103: piece placement for current + 7 previous positions
        # Each position gets 13 channels (12 piece planes + 1 repetition plane)
        for i, hist_board in enumerate(boards):
            base_channel = i * 13

            if hist_board is None:
                continue  # Leave as zeros if no history

            # Determine who is "friendly" based on the CURRENT board's turn
            friendly_color = board.turn

            # Channels 0-5: friendly pieces (pawn, knight, bishop, rook, queen, king)
            # Channels 6-11: enemy pieces
            for square in chess.SQUARES:
                piece = hist_board.piece_at(square)
                if piece:
                    rank = square // 8
                    file = square % 8
                    piece_type = piece.piece_type - 1  # 0-5

                    if piece.color == friendly_color:
                        tensor[rank, file, base_channel + piece_type] = 1
                    else:
                        tensor[rank, file, base_channel + 6 + piece_type] = 1

            # Channel 12: repetition (set to 1 if the position has occurred at least once)
            tensor[:, :, base_channel + 12] = float(hist_board.is_repetition(2)) 

        # Channels 104-107: castling rights
        tensor[:, :, 104] = float(oriented_board.has_queenside_castling_rights(chess.WHITE))
        tensor[:, :, 105] = float(oriented_board.has_kingside_castling_rights(chess.WHITE))
        tensor[:, :, 106] = float(oriented_board.has_queenside_castling_rights(chess.BLACK))
        tensor[:, :, 107] = float(oriented_board.has_kingside_castling_rights(chess.BLACK))

        # Channel 108: side to move (1 if black's turn)
        tensor[:, :, 108] = float(board.turn == chess.BLACK)

        # Channel 109: 50-move rule (halfmove clock / 100, same value across all squares)
        tensor[:, :, 109] = board.halfmove_clock / 100

        # Channel 110: set to 0 (already zeros)

        # Channel 111: set to 1 (helps detect board edges)
        tensor[:, :, 111] = 1

        return tensor

    def output_tensor_to_move(self, network_output: tf.Tensor) -> str:
        """
        Takes a network output and converts it to a move in UIC chess notation
        """
        move_probabilities = network_output[:1858]
        legal_move_probabilities = self.mask_illegal_moves(self.board, move_probabilities)
        best_move_index = tf.argmax(legal_move_probabilities).numpy()

        move_uci = self._get_move_from_index(best_move_index)

        if self.board.turn == chess.BLACK:
            move_uci = self._mirror_move_uci(move_uci)

        return move_uci
    
    def mask_illegal_moves(self, board: chess.Board, move_probabilities: tf.Tensor) -> tf.Tensor:
        """
        Mask illegal moves to remove them from the output and recalculate the probabilities for all legal moves (sum = 1).
        """
        index_lookup = self.index_lookup

        mask = np.zeros(len(move_probabilities), dtype=bool)

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

        # Undo the softmax by converting back to logits
        logits = tf.math.log(move_probabilities)

        # Set illegal moves to -inf
        masked_logits = tf.where(mask, logits, tf.constant(float('-inf')))

        # Re-apply softmax so legal moves sum to 1
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