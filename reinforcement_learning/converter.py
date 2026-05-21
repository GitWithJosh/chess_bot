import chess
import numpy as np
import json
import tensorflow as tf

class Converter:

    def __init__(self):
        self.board = chess.Board
        self.lookup = {}
        with open("move_lookup.json", "r") as f:
            self.lookup = json.load(f)


    def board_to_tensor():
        pass

    def network_to_move(self, network_output: tf.Tensor) -> str:
        """
        Takes a network output and converts it to a move in UIC chess notation
        """
        move_probabilities = network_output[:1858]
        legal_move_probabilities = self.mask_illegal_moves(self.board, move_probabilities)
        best_move_index = tf.argmax(legal_move_probabilities).numpy()
        return self._get_move_from_index(best_move_index)
    
    def mask_illegal_moves(self, board: chess.Board, move_probabilities: tf.Tensor) -> tf.Tensor:
        index_lookup = {v: int(k) for k, v in self.lookup.items()}

        mask = np.zeros(len(move_probabilities), dtype=bool)

        for move in board.legal_moves:
            move_uci = move.uci()
            if move_uci in index_lookup:
                mask[index_lookup[move_uci]] = True

        # Undo the softmax by converting back to logits
        logits = tf.math.log(move_probabilities)

        # Set illegal moves to -inf
        masked_logits = tf.where(mask, logits, tf.constant(float('-inf')))

        # Re-apply softmax so legal moves sum to 1
        return tf.nn.softmax(masked_logits)

    def _get_move_from_index(self, index) -> str:
        """Returns the corresponding move notation given an index"""
        return self.lookup[index]