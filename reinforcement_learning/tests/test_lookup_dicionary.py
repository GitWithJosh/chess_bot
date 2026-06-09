"""Tests for the move lookup dictionary builder."""

import chess
import pytest

from reinforcement_learning.helpers.lookup_dictonary import build_move_lookup


@pytest.fixture(scope="module")
def lookup():
    """Build the lookup once for all tests in this module."""
    return build_move_lookup()


class TestLookupStructure:
    def test_returns_dict(self, lookup):
        assert isinstance(lookup, dict)

    def test_keys_are_sequential_integers(self, lookup):
        keys = sorted(lookup.keys())
        assert keys == list(range(len(lookup)))

    def test_values_are_strings(self, lookup):
        for v in lookup.values():
            assert isinstance(v, str)

    def test_total_count_is_1858(self, lookup):
        """The network expects exactly 1858 moves."""
        assert len(lookup) == 1858


class TestNoDuplicates:
    def test_no_duplicate_moves(self, lookup):
        values = list(lookup.values())
        assert len(values) == len(set(values)), "Duplicate moves found in lookup"


class TestQueenMoves:
    def test_queen_on_a1_has_21_queen_moves(self, lookup):
        """A queen on a1 has 21 moves, plus 2 knight moves from a1 = 23 total."""
        a1_moves = [v for v in lookup.values() if v.startswith("a1")]
        assert len(a1_moves) == 23  # 21 queen + 2 knight

    def test_queen_on_d4_has_moves(self, lookup):
        """d4 has queen moves (27) + knight moves (8) = 35 total."""
        d4_moves = [v for v in lookup.values() if v.startswith("d4")]
        assert len(d4_moves) == 35

    def test_all_queen_moves_valid_squares(self, lookup):
        """All moves should reference valid squares."""
        valid_files = "abcdefgh"
        valid_ranks = "12345678"
        for move in lookup.values():
            assert move[0] in valid_files, f"Invalid from-file in {move}"
            assert move[1] in valid_ranks, f"Invalid from-rank in {move}"
            assert move[2] in valid_files, f"Invalid to-file in {move}"
            assert move[3] in valid_ranks, f"Invalid to-rank in {move}"


class TestKnightMoves:
    def test_knight_on_a1_has_2_moves(self, lookup):
        """Knight on a1 has 2 moves: b3 and c2."""
        # All knight moves from a1
        knight_from_a1 = [v for v in lookup.values() if v == "a1b3" or v == "a1c2"]
        assert len(knight_from_a1) == 2

    def test_knight_on_e4_has_8_moves(self, lookup):
        """Knight on e4 has 8 possible moves."""
        # Knight moves are L-shaped: 2+1 in any direction from e4
        knight_targets = ["d2", "f2", "c3", "g3", "c5", "g5", "d6", "f6"]
        found = [v for v in lookup.values() if v.startswith("e4") and v[2:] in knight_targets]
        assert len(found) == 8


class TestUnderpromotions:
    def test_underpromotions_present(self, lookup):
        """Pawn promotions to knight, bishop, rook should be in the lookup."""
        promotions = [v for v in lookup.values() if len(v) == 5]
        assert len(promotions) > 0

    def test_no_queen_promotions_in_lookup(self, lookup):
        """Queen promotions are implicit (default) and NOT in the lookup."""
        queen_promos = [v for v in lookup.values() if len(v) == 5 and v[4] == "q"]
        assert len(queen_promos) == 0

    def test_all_underpromotion_types(self, lookup):
        """Should have knight, bishop, and rook underpromotions."""
        promo_pieces = set(v[4] for v in lookup.values() if len(v) == 5)
        assert "n" in promo_pieces
        assert "b" in promo_pieces
        assert "r" in promo_pieces


class TestMoveValidity:
    def test_moves_are_pseudo_legal(self, lookup):
        """Each move should be parse-able as a UCI move."""
        for move_str in lookup.values():
            move = chess.Move.from_uci(move_str)
            assert move is not None
            assert move.from_square != move.to_square