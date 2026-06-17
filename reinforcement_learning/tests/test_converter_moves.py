"""Tests for Converter methods: mask_illegal_moves, output_tensor_to_move, _mirror_move_uci."""

import chess
import numpy as np
import pytest
import tensorflow as tf

from reinforcement_learning.helpers.converter import Converter


@pytest.fixture
def converter():
    return Converter()


class TestMirrorMoveUci:
    def test_simple_move(self, converter):
        assert converter._mirror_move_uci("e2e4") == "e7e5"

    def test_a1a8(self, converter):
        assert converter._mirror_move_uci("a1a8") == "a8a1"

    def test_promotion_move(self, converter):
        assert converter._mirror_move_uci("a7a8q") == "a2a1q"

    def test_knight_promotion(self, converter):
        assert converter._mirror_move_uci("b7b8n") == "b2b1n"

    def test_symmetric(self, converter):
        # Mirroring twice should give back the original
        move = "d4d5"
        assert converter._mirror_move_uci(converter._mirror_move_uci(move)) == move

    def test_capture_diagonal(self, converter):
        assert converter._mirror_move_uci("e7d8") == "e2d1"


class TestGetMoveFromIndex:
    def test_valid_index_returns_string(self, converter):
        move = converter._get_move_from_index(0)
        assert isinstance(move, str)
        assert len(move) >= 4

    def test_all_indices_are_valid_uci(self, converter):
        for i in range(min(20, len(converter.lookup))):
            move_str = converter._get_move_from_index(i)
            # Should be a valid UCI string (4 or 5 chars)
            assert 4 <= len(move_str) <= 5

    def test_index_matches_lookup(self, converter):
        for key, val in list(converter.lookup.items())[:10]:
            assert converter._get_move_from_index(int(key)) == val


class TestMaskIllegalMoves:
    def test_output_sums_to_one(self, converter):
        board = chess.Board()
        logits = tf.random.normal([1858])  # raw policy logits (may be negative)
        masked = converter.mask_illegal_moves(board, logits)
        assert abs(float(tf.reduce_sum(masked).numpy()) - 1.0) < 1e-5

    def test_illegal_moves_are_zero(self, converter):
        board = chess.Board()
        logits = tf.random.normal([1858])  # raw policy logits (may be negative)
        masked = converter.mask_illegal_moves(board, logits)

        # Build set of legal move indices
        index_lookup = {v: int(k) for k, v in converter.lookup.items()}
        legal_indices = set()
        for move in board.legal_moves:
            move_uci = move.uci()
            if board.turn == chess.BLACK:
                move_uci = converter._mirror_move_uci(move_uci)
            if move.promotion == chess.QUEEN:
                move_uci = move_uci[:-1]
            if move_uci in index_lookup:
                legal_indices.add(index_lookup[move_uci])

        masked_np = masked.numpy()
        for i in range(1858):
            if i not in legal_indices:
                assert masked_np[i] < 1e-7, f"Illegal move at index {i} has prob {masked_np[i]}"

    def test_only_legal_moves_have_probability(self, converter):
        board = chess.Board()
        logits = tf.random.normal([1858])  # raw policy logits (may be negative)
        masked = converter.mask_illegal_moves(board, logits)
        masked_np = masked.numpy()

        # Count non-zero entries
        nonzero_count = np.sum(masked_np > 1e-7)
        legal_move_count = len(list(board.legal_moves))
        assert nonzero_count == legal_move_count

    def test_works_for_black(self, converter):
        board = chess.Board()
        board.push_uci("e2e4")  # Now black to move
        logits = tf.random.normal([1858])  # raw policy logits (may be negative)
        masked = converter.mask_illegal_moves(board, logits)
        assert abs(float(tf.reduce_sum(masked).numpy()) - 1.0) < 1e-5


class TestOutputTensorToMove:
    def test_returns_legal_move(self, converter):
        board = chess.Board()
        converter.board = board
        # Create a fake network output with uniform distribution
        network_output = tf.random.normal([1858])  # raw policy logits
        move_uci = converter.output_tensor_to_move(network_output)
        move = chess.Move.from_uci(move_uci)
        assert move in board.legal_moves

    def test_returns_legal_move_for_black(self, converter):
        board = chess.Board()
        board.push_uci("e2e4")
        converter.board = board
        network_output = tf.random.normal([1858])  # raw policy logits
        move_uci = converter.output_tensor_to_move(network_output)
        move = chess.Move.from_uci(move_uci)
        assert move in board.legal_moves

    def test_returns_string(self, converter):
        board = chess.Board()
        converter.board = board
        network_output = tf.random.normal([1858])  # raw policy logits
        move_uci = converter.output_tensor_to_move(network_output)
        assert isinstance(move_uci, str)
        assert 4 <= len(move_uci) <= 5