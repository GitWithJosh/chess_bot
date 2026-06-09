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
        assert tensor.shape == (8, 8, 112)

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
        assert tensor[0, 0, 104] == 1  # White queenside
        assert tensor[0, 0, 105] == 1  # White kingside
        assert tensor[0, 0, 106] == 1  # Black queenside
        assert tensor[0, 0, 107] == 1  # Black kingside

    def test_no_castling_rights(self, converter):
        """Position with no castling rights"""
        board = chess.Board()
        board.set_castling_fen("-")
        tensor = converter.board_to_input_tensor(board)
        assert tensor[0, 0, 104] == 0
        assert tensor[0, 0, 105] == 0
        assert tensor[0, 0, 106] == 0
        assert tensor[0, 0, 107] == 0

    def test_castling_fills_entire_plane(self, converter):
        """Castling rights should be the same value across all squares"""
        board = chess.Board()
        tensor = converter.board_to_input_tensor(board)
        for ch in [104, 105, 106, 107]:
            assert np.all(tensor[:, :, ch] == tensor[0, 0, ch])


class TestSideToMove:
    def test_white_to_move(self, converter):
        board = chess.Board()
        tensor = converter.board_to_input_tensor(board)
        assert tensor[0, 0, 108] == 0  # White's turn -> 0

    def test_black_to_move(self, converter):
        board = chess.Board()
        board.push_uci("e2e4")
        tensor = converter.board_to_input_tensor(board)
        assert tensor[0, 0, 108] == 1  # Black's turn -> 1


class TestFiftyMoveRule:
    def test_starting_position_zero(self, converter):
        board = chess.Board()
        tensor = converter.board_to_input_tensor(board)
        assert tensor[0, 0, 109] == 0

    def test_after_knight_moves(self, converter):
        """Knight moves increment halfmove clock"""
        board = chess.Board()
        board.push_uci("g1f3")  # halfmove_clock = 1
        board.push_uci("g8f6")  # halfmove_clock = 2
        tensor = converter.board_to_input_tensor(board)
        expected = 2 / 99.0
        assert abs(float(tensor[0, 0, 109]) - expected) < 1e-3

    def test_fills_entire_plane(self, converter):
        board = chess.Board()
        board.push_uci("g1f3")
        tensor = converter.board_to_input_tensor(board)
        assert np.all(tensor[:, :, 109] == tensor[0, 0, 109])


class TestConstantPlanes:
    def test_channel_110_zeros(self, converter):
        board = chess.Board()
        tensor = converter.board_to_input_tensor(board)
        assert np.all(tensor[:, :, 110] == 0)

    def test_channel_111_ones(self, converter):
        board = chess.Board()
        tensor = converter.board_to_input_tensor(board)
        assert np.all(tensor[:, :, 111] == 1)


class TestMoveHistory:
    def test_history_populates_previous_planes(self, converter):
        """After one move, channels 13-25 should have the starting position"""
        board = chess.Board()
        board.push_uci("e2e4")
        tensor = converter.board_to_input_tensor(board)

        # It's black's turn, board is mirrored.
        # On the mirrored board, friendly_color = BLACK (board.turn).
        # The mirrored board swaps colors+ranks, so original black pawns (rank 6)
        # become WHITE pawns at rank 1 on the mirrored board.
        # Since friendly_color=BLACK, those WHITE pawns are "enemy" -> ch 6.
        # Original white e4 pawn becomes BLACK pawn at rank 4 -> "friendly" -> ch 0.
        # Current position (ch 0-12):
        assert tensor[4, 4, 0] == 1  # e4 pawn is BLACK on mirrored board -> friendly (ch 0)

        # Previous position (ch 13-25): starting position mirrored
        # Original white e2 pawn -> mirrored to BLACK at rank 6 -> friendly -> ch 13+0=13
        assert tensor[6, 4, 13] == 1

    def test_no_history_is_zeros(self, converter):
        """Starting position has no history, so channels 13-103 should be zero for pieces"""
        board = chess.Board()
        tensor = converter.board_to_input_tensor(board)
        for i in range(1, 8):
            base = i * 13
            # Piece planes should be zero (no history)
            assert np.sum(tensor[:, :, base:base + 12]) == 0


class TestRepetition:
    def test_no_repetition_starting(self, converter):
        """Starting position with no moves has no repetition"""
        board = chess.Board()
        tensor = converter.board_to_input_tensor(board)
        print(tensor[0,0,12])
        assert tensor[0, 0, 12] == 0

    def test_repetition_detected(self, converter):
        """After Nf3-Ng1-Nf3, the position has repeated"""
        board = chess.Board()
        board.push_uci("g1f3")
        board.push_uci("g8f6")
        board.push_uci("f3g1")
        board.push_uci("f6g8")
        # Now back to starting position — this is a repetition
        tensor = converter.board_to_input_tensor(board)
        assert tensor[0, 0, 12] == 1

    def test_repetition_fills_plane(self, converter):
        """Repetition value should be the same across all squares"""
        board = chess.Board()
        board.push_uci("g1f3")
        board.push_uci("g8f6")
        board.push_uci("f3g1")
        board.push_uci("f6g8")
        tensor = converter.board_to_input_tensor(board)
        assert np.all(tensor[:, :, 12] == tensor[0, 0, 12])


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