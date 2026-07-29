"""Tests for MCTS search, backpropagation, and SelfPlayGame."""

import chess
import numpy as np
import pytest

from reinforcement_learning.monte_carlo_tree_search.mcts_v2 import MCTS, SelfPlayGame
from reinforcement_learning.monte_carlo_tree_search.nodes_and_edges_v2 import Node, Edge
from reinforcement_learning.helpers.converter import Converter


class MockNetwork:
    """A mock network that returns uniform policy and neutral value."""

    def predict(self, board_tensor):
        policy = np.ones(1858, dtype=np.float32) / 1858
        value = np.array([0.0])
        return policy, value


class MockNetworkWDL:
    """A mock network that returns WDL value."""

    def predict(self, board_tensor):
        policy = np.ones(1858, dtype=np.float32) / 1858
        value = np.array([0.5, 0.3, 0.2])  # win=0.5, draw=0.3, loss=0.2
        return policy, value


@pytest.fixture
def converter():
    return Converter()


@pytest.fixture
def mcts(converter):
    return MCTS(
        network=MockNetwork(),
        converter=converter,
        num_simulations=10,
        c_puct=1.5,
    )


@pytest.fixture
def mcts_wdl(converter):
    return MCTS(
        network=MockNetworkWDL(),
        converter=converter,
        num_simulations=10,
        c_puct=1.5,
    )


# ---------------------------------------------------------------------------
# MCTS Search
# ---------------------------------------------------------------------------

class TestMCTSSearch:
    def test_search_returns_root(self, mcts):
        board = chess.Board()
        root = Node(board)
        result = mcts.search(root, add_noise=False)
        assert result is root

    def test_search_expands_root(self, mcts):
        board = chess.Board()
        root = Node(board)
        mcts.search(root, add_noise=False)
        assert not root.is_leaf
        assert len(root.edges) > 0

    def test_search_accumulates_visits(self, mcts):
        board = chess.Board()
        root = Node(board)
        mcts.search(root, add_noise=False)
        total_visits = root.total_visits
        # Should have at least num_simulations visits
        assert total_visits >= mcts.num_simulations

    def test_search_with_noise(self, mcts):
        board = chess.Board()
        root = Node(board)
        mcts.search(root, add_noise=True)
        assert root.total_visits >= mcts.num_simulations

    def test_search_wdl_network(self, mcts_wdl):
        board = chess.Board()
        root = Node(board)
        mcts_wdl.search(root, add_noise=False)
        assert root.total_visits >= mcts_wdl.num_simulations


# ---------------------------------------------------------------------------
# Backpropagation
# ---------------------------------------------------------------------------

class TestBackpropagate:
    def test_single_edge(self, mcts):
        board = chess.Board()
        node = Node(board)
        edge = Edge(chess.Move.from_uci("e2e4"), node, prior=0.5)

        mcts._backpropagate([edge], value=0.8)
        # Value is negated once (parent perspective)
        assert edge.N == 1
        assert np.isclose(edge.W, -0.8)

    def test_two_edges(self, mcts):
        board = chess.Board()
        node = Node(board)
        e1 = Edge(chess.Move.from_uci("e2e4"), node, prior=0.5)
        e2 = Edge(chess.Move.from_uci("d2d4"), node, prior=0.5)

        mcts._backpropagate([e1, e2], value=0.6)
        # e2 (last in path, closest to expanded node): value = -0.6
        # e1 (first in path): value = -(-0.6) = 0.6
        assert np.isclose(e2.W, -0.6)
        assert np.isclose(e1.W, 0.6)


# ---------------------------------------------------------------------------
# Dirichlet Noise
# ---------------------------------------------------------------------------

class TestDirichletNoise:
    def test_noise_changes_priors(self, mcts, converter):
        board = chess.Board()
        root = Node(board)
        root.expand(MockNetwork(), converter)

        original_priors = [e.P for e in root.edges]
        mcts._add_dirichlet_noise(root)
        new_priors = [e.P for e in root.edges]

        # At least some priors should have changed
        changed = sum(1 for o, n in zip(original_priors, new_priors) if not np.isclose(o, n))
        assert changed > 0

    def test_priors_still_sum_to_approx_one(self, mcts, converter):
        board = chess.Board()
        root = Node(board)
        root.expand(MockNetwork(), converter)
        mcts._add_dirichlet_noise(root)

        prior_sum = sum(e.P for e in root.edges)
        assert np.isclose(prior_sum, 1.0, atol=0.01)


# ---------------------------------------------------------------------------
# Get Best Move
# ---------------------------------------------------------------------------

class TestGetBestMove:
    def test_greedy_returns_most_visited(self, mcts, converter):
        board = chess.Board()
        root = Node(board)
        root.expand(MockNetwork(), converter)

        # Set one edge to have many more visits
        root.edges[0].N = 100
        for e in root.edges[1:]:
            e.N = 1

        best = mcts.get_best_move(root, temperature=0)
        assert best == root.edges[0].move

    def test_returns_valid_move(self, mcts):
        board = chess.Board()
        root = Node(board)
        mcts.search(root, add_noise=False)
        move = mcts.get_best_move(root, temperature=0)
        assert move in board.legal_moves

    def test_temperature_one_returns_valid(self, mcts):
        board = chess.Board()
        root = Node(board)
        mcts.search(root, add_noise=False)
        move = mcts.get_best_move(root, temperature=1.0)
        assert move in board.legal_moves


# ---------------------------------------------------------------------------
# Reuse Subtree
# ---------------------------------------------------------------------------

class TestReuseSubtree:
    def test_reuse_known_move(self, mcts, converter):
        board = chess.Board()
        root = Node(board)
        root.expand(MockNetwork(), converter)

        # Get first edge's move
        move = root.edges[0].move
        # Create the child
        root.get_child_node(root.edges[0])

        new_root = mcts.reuse_subtree(root, move)
        assert new_root is not None
        assert new_root.board.turn != board.turn

    def test_reuse_unknown_move_creates_new(self, mcts, converter):
        board = chess.Board()
        root = Node(board)
        # Don't expand - so no edges

        move = chess.Move.from_uci("e2e4")
        new_root = mcts.reuse_subtree(root, move)
        assert new_root is not None
        # The board should have the move applied
        assert new_root.board.turn == chess.BLACK


# ---------------------------------------------------------------------------
# SelfPlayGame
# ---------------------------------------------------------------------------

class TestSelfPlayGame:
    def test_play_returns_training_data(self, converter):
        network = MockNetwork()
        mcts = MCTS(network=network, converter=converter, num_simulations=2)
        game = SelfPlayGame(mcts, temperature_threshold=5, max_moves=10)
        data = game.play()

        assert isinstance(data, list)
        assert len(data) > 0

    def test_training_data_has_correct_keys(self, converter):
        network = MockNetwork()
        mcts = MCTS(network=network, converter=converter, num_simulations=2)
        game = SelfPlayGame(mcts, temperature_threshold=5, max_moves=10)
        data = game.play()

        for sample in data:
            assert "board_tensor" in sample
            assert "policy_target" in sample
            assert "value_target" in sample

    def test_training_data_shapes(self, converter):
        network = MockNetwork()
        mcts = MCTS(network=network, converter=converter, num_simulations=2)
        game = SelfPlayGame(mcts, temperature_threshold=5, max_moves=10)
        data = game.play()

        for sample in data:
            assert sample["board_tensor"].shape == (8, 8, 20)
            assert sample["policy_target"].shape == (1858,)
            assert isinstance(sample["value_target"], float)

    def test_value_targets_are_valid(self, converter):
        network = MockNetwork()
        mcts = MCTS(network=network, converter=converter, num_simulations=2)
        game = SelfPlayGame(mcts, temperature_threshold=5, max_moves=10)
        data = game.play()

        for sample in data:
            assert sample["value_target"] in [-1.0, 0.0, 1.0]

    def test_max_moves_respected(self, converter):
        network = MockNetwork()
        mcts = MCTS(network=network, converter=converter, num_simulations=2)
        game = SelfPlayGame(mcts, temperature_threshold=5, max_moves=5)
        data = game.play()
        assert len(data) <= 5

    def test_get_game_result_draw(self, converter):
        network = MockNetwork()
        mcts = MCTS(network=network, converter=converter, num_simulations=2)
        game = SelfPlayGame(mcts, max_moves=10)

        board = chess.Board()
        result = game._get_game_result(board, 10)
        assert result == "1/2-1/2"

    def test_assign_values_white_wins(self, converter):
        network = MockNetwork()
        mcts = MCTS(network=network, converter=converter, num_simulations=2)
        game = SelfPlayGame(mcts)

        training_data = [
            {"side_to_move": chess.WHITE, "board_tensor": None, "policy_target": None},
            {"side_to_move": chess.BLACK, "board_tensor": None, "policy_target": None},
        ]

        result = game._assign_values(training_data, "1-0")
        assert result[0]["value_target"] == 1.0
        assert result[1]["value_target"] == -1.0

    def test_assign_values_draw(self, converter):
        network = MockNetwork()
        mcts = MCTS(network=network, converter=converter, num_simulations=2)
        game = SelfPlayGame(mcts)

        training_data = [
            {"side_to_move": chess.WHITE, "board_tensor": None, "policy_target": None},
            {"side_to_move": chess.BLACK, "board_tensor": None, "policy_target": None},
        ]

        result = game._assign_values(training_data, "1/2-1/2")
        assert result[0]["value_target"] == 0.0
        assert result[1]["value_target"] == 0.0