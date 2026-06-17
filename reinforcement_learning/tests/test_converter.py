import chess
import numpy as np
import pytest

from reinforcement_learning.helpers.converter import Converter

@pytest.fixture
def converter():
    return Converter()


class TestShape:
    def test_output_shape(self, converter):
        board = chess.Board()
        tensor = converter.board_to_input_tensor(board)
        assert tensor.shape == (8, 8, 20)

    def test_dtype(self, converter):
        board = chess.Board()
        tensor = converter.board_to_input_tensor(board)
        assert tensor.dtype == np.float16


class TestStartingPosition:
    def test_white_pawns(self, converter):
        """Starting position: white pawns on rank 1 (index 1), channels 0 (friendly pawn)"""
        board = chess.Board()
        tensor = converter.board_to_input_tensor(board)
        # White to move, so white = friendly. Pawns = piece_type 0.
        for file in range(8):
            assert tensor[1, file, 0] == 1, f"White pawn missing at file {file}"

    def test_black_pawns(self, converter):
        """Black pawns on rank 6, channels 6 (enemy pawn)"""
        board = chess.Board()
        tensor = converter.board_to_input_tensor(board)
        for file in range(8):
            assert tensor[6, file, 6] == 1, f"Black pawn missing at file {file}"

    def test_white_king(self, converter):
        """White king on e1 (rank 0, file 4), channel 5 (friendly king)"""
        board = chess.Board()
        tensor = converter.board_to_input_tensor(board)
        assert tensor[0, 4, 5] == 1

    def test_black_king(self, converter):
        """Black king on e8 (rank 7, file 4), channel 11 (enemy king)"""
        board = chess.Board()
        tensor = converter.board_to_input_tensor(board)
        assert tensor[7, 4, 11] == 1

    def test_empty_squares_are_zero(self, converter):
        """Middle of the board should have no pieces in starting position"""
        board = chess.Board()
        tensor = converter.board_to_input_tensor(board)
        for rank in range(2, 6):
            for file in range(8):
                assert np.sum(tensor[rank, file, 0:12]) == 0


class TestCastlingRights:
    def test_starting_position_all_castling(self, converter):
        """Starting position has all castling rights"""
        board = chess.Board()
        tensor = converter.board_to_input_tensor(board)
        assert tensor[0, 0, 12] == 1  # Friendly queenside
        assert tensor[0, 0, 13] == 1  # Friendly kingside
        assert tensor[0, 0, 14] == 1  # Enemy queenside
        assert tensor[0, 0, 15] == 1  # Enemy kingside

    def test_no_castling_rights(self, converter):
        """Position with no castling rights"""
        board = chess.Board()
        board.set_castling_fen("-")
        tensor = converter.board_to_input_tensor(board)
        assert tensor[0, 0, 12] == 0
        assert tensor[0, 0, 13] == 0
        assert tensor[0, 0, 14] == 0
        assert tensor[0, 0, 15] == 0

    def test_castling_fills_entire_plane(self, converter):
        """Castling rights should be the same value across all squares"""
        board = chess.Board()
        tensor = converter.board_to_input_tensor(board)
        for ch in [12, 13, 14, 15]:
            assert np.all(tensor[:, :, ch] == tensor[0, 0, ch])


class TestSideToMove:
    def test_white_to_move(self, converter):
        board = chess.Board()
        tensor = converter.board_to_input_tensor(board)
        assert tensor[0, 0, 16] == 0  # White's turn -> 0

    def test_black_to_move(self, converter):
        board = chess.Board()
        board.push_uci("e2e4")
        tensor = converter.board_to_input_tensor(board)
        assert tensor[0, 0, 16] == 1  # Black's turn -> 1


class TestFiftyMoveRule:
    def test_starting_position_zero(self, converter):
        board = chess.Board()
        tensor = converter.board_to_input_tensor(board)
        assert tensor[0, 0, 17] == 0

    def test_after_knight_moves(self, converter):
        """Knight moves increment halfmove clock"""
        board = chess.Board()
        board.push_uci("g1f3")  # halfmove_clock = 1
        board.push_uci("g8f6")  # halfmove_clock = 2
        tensor = converter.board_to_input_tensor(board)
        expected = 2 / 100.0
        assert abs(float(tensor[0, 0, 17]) - expected) < 1e-3

    def test_fills_entire_plane(self, converter):
        board = chess.Board()
        board.push_uci("g1f3")
        tensor = converter.board_to_input_tensor(board)
        assert np.all(tensor[:, :, 17] == tensor[0, 0, 17])


class TestEnPassant:
    def test_no_en_passant_starting(self, converter):
        """Starting position has no en passant target"""
        board = chess.Board()
        tensor = converter.board_to_input_tensor(board)
        assert np.all(tensor[:, :, 18] == 0)

    def test_double_push_without_capturer_is_zero(self, converter):
        """1.e4 sets an ep square in the FEN but no pawn can capture, so the
        plane stays empty (we gate on has_legal_en_passant())."""
        board = chess.Board()
        board.push_uci("e2e4")
        tensor = converter.board_to_input_tensor(board)
        assert np.all(tensor[:, :, 18] == 0)

    def test_legal_en_passant_marks_target_square(self, converter):
        """1.e4 Nf6 2.e5 d5 — white can play exd6 e.p.; d6 is the target.
        White to move, so the board is not mirrored: d6 = (rank 5, file 3)."""
        board = chess.Board()
        for mv in ("e2e4", "g8f6", "e4e5", "d7d5"):
            board.push_uci(mv)
        tensor = converter.board_to_input_tensor(board)
        assert tensor[5, 3, 18] == 1
        assert np.sum(tensor[:, :, 18]) == 1  # exactly one square marked


class TestConstantPlanes:
    def test_channel_19_ones(self, converter):
        board = chess.Board()
        tensor = converter.board_to_input_tensor(board)
        assert np.all(tensor[:, :, 19] == 1)


class TestFriendlyEnemyPerspective:
    def test_perspective_flips_on_black_turn(self, converter):
        """When it's black's turn, board is mirrored. friendly_color = board.turn = BLACK.
        On mirrored board: original black pawns become WHITE, original white become BLACK.
        So BLACK pieces on mirrored board (original white) are "friendly" (ch 0-5)."""
        board = chess.Board()
        board.push_uci("e2e4")
        tensor = converter.board_to_input_tensor(board)

        # On mirrored board, original black pawns at rank 6 -> WHITE at rank 1
        # friendly_color=BLACK, so WHITE pawns at rank 1 are enemy (ch 6)
        assert tensor[1, 0, 6] == 1  # enemy pawn at rank 1

        # On mirrored board, original white e4 pawn -> BLACK at rank 4 (e5 mirror)
        # friendly_color=BLACK, so BLACK pawn is friendly (ch 0)
        assert tensor[4, 4, 0] == 1  # friendly pawn at rank 4

if __name__ == "__main__":
    pytest.main([__file__, "-v"])