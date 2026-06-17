"""Tests for the AlphaZero 73-plane policy move encoding / gather table."""

import json

import numpy as np
import pytest

from helpers.move_encoding import (
    NUM_PLANES,
    build_gather_indices,
    gather_indices_from_lookup_path,
    move_uci_to_plane,
)

LOOKUP_PATH = "reinforcement_learning/move_lookup.json"


@pytest.fixture(scope="module")
def lookup():
    with open(LOOKUP_PATH) as f:
        return json.load(f)


class TestGatherTable:
    def test_bijection_over_full_lookup(self, lookup):
        """Every move maps to a distinct cell; build raises on collisions."""
        idx = build_gather_indices(lookup)
        assert idx.shape == (len(lookup),)
        assert idx.dtype == np.int32
        assert len(set(idx.tolist())) == len(lookup)  # all distinct

    def test_indices_in_range(self, lookup):
        idx = build_gather_indices(lookup)
        assert idx.min() >= 0
        assert idx.max() < 8 * 8 * NUM_PLANES

    def test_plane_group_counts(self, lookup):
        """1456 sliding + 336 knight + 66 under-promotion = 1858 (chess-geometry constants)."""
        planes = np.array([move_uci_to_plane(u) for u in lookup.values()])
        assert int(((planes >= 0) & (planes < 56)).sum()) == 1456
        assert int(((planes >= 56) & (planes < 64)).sum()) == 336
        assert int(((planes >= 64) & (planes < 73)).sum()) == 66

    def test_from_path_matches_dict(self, lookup):
        a = gather_indices_from_lookup_path(LOOKUP_PATH)
        b = build_gather_indices(lookup)
        assert np.array_equal(a, b)


class TestMovePlaneEncoding:
    def test_sliding_north(self):
        # a1a8: direction N (dir 0), distance 7 -> 0*7 + 6
        assert move_uci_to_plane("a1a8") == 6

    def test_sliding_diagonal(self):
        # a1h8: direction NE (dir 1), distance 7 -> 1*7 + 6
        assert move_uci_to_plane("a1h8") == 13

    def test_castling_is_sliding(self):
        # e1g1: king two squares east (dir 2 = E), distance 2 -> 2*7 + 1
        assert move_uci_to_plane("e1g1") == 15

    def test_knight_plane_range(self):
        for uci in ("b1c3", "b1a3", "g1f3", "g1h3"):
            p = move_uci_to_plane(uci)
            assert 56 <= p < 64

    def test_underpromotion_plane_range(self):
        for uci in ("a7a8n", "a7a8b", "a7a8r", "a7b8n", "h7g8r"):
            p = move_uci_to_plane(uci)
            assert 64 <= p < 73

    def test_queen_promotion_is_sliding_not_underpromo(self):
        # 4-char promotion (queen) must land in the sliding planes, not 64-72
        assert move_uci_to_plane("a7a8") < 56

    def test_rejects_non_axis_aligned(self):
        with pytest.raises(ValueError):
            move_uci_to_plane("a1b4")  # not a knight, not axis-aligned


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
