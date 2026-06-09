import os
import numpy as np

from reinforcement_learning.networks.big_network import BigNetwork


def test_predict_shapes_and_probabilities():
    net = BigNetwork(num_res_blocks=1, num_filters=8, learning_rate=0.01)
    board = np.random.rand(8, 8, 112).astype(np.float32)
    policy, value = net.predict(board)

    assert policy.shape == (net.num_moves,)
    assert value.shape == (3,)
    assert np.all(policy >= 0)
    assert np.isclose(policy.sum(), 1.0, atol=1e-5)
    assert np.isclose(value.sum(), 1.0, atol=1e-5)


def test_train_one_epoch(tmp_path):
    net = BigNetwork(num_res_blocks=1, num_filters=8, learning_rate=0.01)
    batch_size = 2
    boards = np.random.rand(batch_size, 8, 8, 112).astype(np.float32)

    policies = np.random.rand(batch_size, net.num_moves).astype(np.float32)
    policies = policies / policies.sum(axis=1, keepdims=True)

    # BigNetwork.train() calls outcome_to_wdl() internally, so pass raw outcomes
    values = np.array([1, -1], dtype=np.float32)

    history = net.train(boards, policies, values, epochs=1, batch_size=1, verbose=0)
    assert hasattr(history, "history")
    assert "loss" in history.history


def test_save_and_load(tmp_path):
    net = BigNetwork(num_res_blocks=1, num_filters=8, learning_rate=0.01)
    path = str(tmp_path / "weights.weights.h5")
    net.save(path)
    assert os.path.exists(path)

    net2 = BigNetwork(num_res_blocks=1, num_filters=8, learning_rate=0.01)
    net2.load(path)