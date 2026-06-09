"""Tests for weights_utils: extract_iteration and get_latest_weights."""

import os
import pytest

from reinforcement_learning.helpers.weights_utils import extract_iteration, get_latest_weights


class TestExtractIteration:
    def test_simple_version(self):
        assert extract_iteration("small_network_v0.weights.h5") == 0

    def test_higher_version(self):
        assert extract_iteration("small_network_v10.weights.h5") == 10

    def test_with_full_path(self):
        assert extract_iteration("/full/path/to/small_network_v42.weights.h5") == 42

    def test_big_network(self):
        assert extract_iteration("big_network_v3.weights.h5") == 3

    def test_no_version_returns_minus_one(self):
        assert extract_iteration("random_model.weights.h5") == -1

    def test_no_v_prefix(self):
        assert extract_iteration("network_12.weights.h5") == -1

    def test_empty_string(self):
        assert extract_iteration("") == -1

    def test_multiple_v_numbers(self):
        # Should find the first match
        result = extract_iteration("v2_network_v5.weights.h5")
        assert result == 2


class TestGetLatestWeights:
    def test_finds_latest_by_iteration(self, tmp_path):
        # Create fake weight files
        (tmp_path / "small_network_v0.weights.h5").touch()
        (tmp_path / "small_network_v1.weights.h5").touch()
        (tmp_path / "small_network_v5.weights.h5").touch()

        result = get_latest_weights("small", str(tmp_path))
        assert result == str(tmp_path / "small_network_v5.weights.h5")

    def test_filters_by_network_prefix(self, tmp_path):
        (tmp_path / "small_network_v10.weights.h5").touch()
        (tmp_path / "big_network_v2.weights.h5").touch()

        result = get_latest_weights("big", str(tmp_path))
        assert result == str(tmp_path / "big_network_v2.weights.h5")

    def test_falls_back_to_all_files(self, tmp_path):
        # No matching prefix, should fall back to any weights file
        (tmp_path / "random_model.weights.h5").touch()

        result = get_latest_weights("nonexistent", str(tmp_path))
        assert result == str(tmp_path / "random_model.weights.h5")

    def test_raises_on_missing_directory(self):
        with pytest.raises(FileNotFoundError, match="Weights directory not found"):
            get_latest_weights("small", "/nonexistent/path/that/does/not/exist")

    def test_raises_on_empty_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No weights found"):
            get_latest_weights("small", str(tmp_path))

    def test_ignores_non_h5_files(self, tmp_path):
        (tmp_path / "small_network_v5.weights.h5").touch()
        (tmp_path / "small_network_v99.txt").touch()  # Not a weights file

        result = get_latest_weights("small", str(tmp_path))
        assert result == str(tmp_path / "small_network_v5.weights.h5")

    def test_returns_full_path(self, tmp_path):
        (tmp_path / "small_network_v0.weights.h5").touch()

        result = get_latest_weights("small", str(tmp_path))
        assert os.path.isabs(result)