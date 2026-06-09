"""Additional tests for BigNetwork: outcome_to_wdl and architectural properties."""

import numpy as np
import pytest

from reinforcement_learning.networks.big_network import BigNetwork


@pytest.fixture(scope="module")
def big_net():
    return BigNetwork(num_res_blocks=1, num_filters=8, learning_rate=0.01)


class TestOutcomeToWdl:
    def test_win(self, big_net):
        outcomes = np.array([1])
        wdl = big_net.outcome_to_wdl(outcomes)
        np.testing.assert_array_equal(wdl, [[1.0, 0.0, 0.0]])

    def test_draw(self, big_net):
        outcomes = np.array([0])
        wdl = big_net.outcome_to_wdl(outcomes)
        np.testing.assert_array_equal(wdl, [[0.0, 1.0, 0.0]])

    def test_loss(self, big_net):
        outcomes = np.array([-1])
        wdl = big_net.outcome_to_wdl(outcomes)
        np.testing.assert_array_equal(wdl, [[0.0, 0.0, 1.0]])

    def test_batch(self, big_net):
        outcomes = np.array([1, 0, -1, 1, -1])
        wdl = big_net.outcome_to_wdl(outcomes)
        assert wdl.shape == (5, 3)
        np.testing.assert_array_equal(wdl[0], [1, 0, 0])
        np.testing.assert_array_equal(wdl[1], [0, 1, 0])
        np.testing.assert_array_equal(wdl[2], [0, 0, 1])

    def test_2d_input(self, big_net):
        outcomes = np.array([[1], [0], [-1]])
        wdl = big_net.outcome_to_wdl(outcomes)
        assert wdl.shape == (3, 3)
        np.testing.assert_array_equal(wdl[0], [1, 0, 0])

    def test_rows_sum_to_one(self, big_net):
        outcomes = np.array([1, 0, -1, 1, 0])
        wdl = big_net.outcome_to_wdl(outcomes)
        assert np.all(wdl.sum(axis=1) == 1.0)


class TestBigNetArchitecture:
    def test_has_two_outputs(self, big_net):
        assert len(big_net.model.outputs) == 2

    def test_policy_output_shape(self, big_net):
        policy_layer = big_net.model.get_layer("policy_output")
        assert policy_layer.output.shape[-1] == 1858

    def test_value_output_shape_wdl(self, big_net):
        value_layer = big_net.model.get_layer("value_output")
        assert value_layer.output.shape[-1] == 3

    def test_value_output_sums_to_one(self, big_net):
        board = np.random.rand(8, 8, 112).astype(np.float32)
        _, value = big_net.predict(board)
        assert np.isclose(value.sum(), 1.0, atol=1e-5)


class TestBigNetTrain:
    def test_train_with_outcome_conversion(self):
        """Test that training works with raw outcomes that get converted to WDL."""
        net = BigNetwork(num_res_blocks=1, num_filters=8, learning_rate=0.01)
        batch_size = 4
        boards = np.random.rand(batch_size, 8, 8, 112).astype(np.float32)
        policies = np.random.rand(batch_size, 1858).astype(np.float32)
        policies = policies / policies.sum(axis=1, keepdims=True)
        # Raw outcomes: 1, -1, 0, 1
        values = np.array([1, -1, 0, 1], dtype=np.float32)

        history = net.train(boards, policies, values, epochs=1, batch_size=4, verbose=0)
        assert "loss" in history.history