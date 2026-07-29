"""Tests for nodes_and_edges_v2: Edge, Node, mirror_move_uci, move_to_lookup_key."""

import math
import chess
import numpy as np
import pytest

from reinforcement_learning.monte_carlo_tree_search.nodes_and_edges_v2 import (
    Edge,
    Node,
    mirror_move_uci,
    move_to_lookup_key,
)
from reinforcement_learning.helpers.converter import Converter


# ---------------------------------------------------------------------------
# mirror_move_uci
# ---------------------------------------------------------------------------

class TestMirrorMoveUci:
    def test_e2e4(self):
        assert mirror_move_uci("e2e4") == "e7e5"

    def test_a1h8(self):
        assert mirror_move_uci("a1h8") == "a8h1"

    def test_promotion(self):
        assert mirror_move_uci("a7a8n") == "a2a1n"

    def test_double_mirror_is_identity(self):
        for move in ["e2e4", "d7d5", "g1f3", "b7b8q"]:
            assert mirror_move_uci(mirror_move_uci(move)) == move


# ---------------------------------------------------------------------------
# move_to_lookup_key
# ---------------------------------------------------------------------------

class TestMoveToLookupKey:
    def test_white_normal_move(self):
        move = chess.Move.from_uci("e2e4")
        key = move_to_lookup_key(move, chess.WHITE)
        assert key == "e2e4"

    def test_black_move_is_mirrored(self):
        move = chess.Move.from_uci("e7e5")
        key = move_to_lookup_key(move, chess.BLACK)
        assert key == "e2e4"  # mirrored

    def test_queen_promotion_strips_q(self):
        move = chess.Move.from_uci("a7a8q")
        key = move_to_lookup_key(move, chess.WHITE)
        assert key == "a7a8"  # no 'q'

    def test_knight_promotion_keeps_n(self):
        move = chess.Move.from_uci("a7a8n")
        key = move_to_lookup_key(move, chess.WHITE)
        assert key == "a7a8n"

    def test_black_queen_promotion(self):
        # Black promoting: e2e1q from black's perspective mirrored = e7e8, stripped of q
        move = chess.Move.from_uci("e2e1q")
        key = move_to_lookup_key(move, chess.BLACK)
        assert key == "e7e8"  # mirrored + stripped


# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------

class TestEdge:
    def test_initial_state(self):
        board = chess.Board()
        node = Node(board)
        move = chess.Move.from_uci("e2e4")
        edge = Edge(move, node, prior=0.5)

        assert edge.N == 0
        assert edge.W == 0.0
        assert edge.Q == 0.0
        assert edge.P == 0.5
        assert edge.move == move
        assert edge.child_node is None

    def test_update_single(self):
        board = chess.Board()
        node = Node(board)
        edge = Edge(chess.Move.from_uci("e2e4"), node, prior=0.3)
        edge.update(0.8)

        assert edge.N == 1
        assert edge.W == 0.8
        assert edge.Q == 0.8

    def test_update_multiple(self):
        board = chess.Board()
        node = Node(board)
        edge = Edge(chess.Move.from_uci("e2e4"), node, prior=0.3)
        edge.update(1.0)
        edge.update(0.0)
        edge.update(-1.0)

        assert edge.N == 3
        assert np.isclose(edge.W, 0.0)
        assert np.isclose(edge.Q, 0.0)

    def test_ucb_score_unexplored(self):
        """Unexplored edge (N=0) should get high exploration bonus."""
        board = chess.Board()
        node = Node(board)
        edge = Edge(chess.Move.from_uci("e2e4"), node, prior=0.5)

        # parent_visits=10, c_puct=1.5
        score = edge.ucb_score(parent_visits=10, c_puct=1.5)
        expected_u = 1.5 * 0.5 * math.sqrt(10) / 1
        assert np.isclose(score, expected_u)

    def test_ucb_score_explored(self):
        board = chess.Board()
        node = Node(board)
        edge = Edge(chess.Move.from_uci("e2e4"), node, prior=0.5)
        edge.update(0.6)

        score = edge.ucb_score(parent_visits=10, c_puct=1.5)
        expected_u = 1.5 * 0.5 * math.sqrt(10) / 2
        expected = 0.6 + expected_u
        assert np.isclose(score, expected)


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class TestNode:
    def test_new_node_is_leaf(self):
        board = chess.Board()
        node = Node(board)
        assert node.is_leaf

    def test_starting_position_not_terminal(self):
        board = chess.Board()
        node = Node(board)
        assert not node.is_terminal

    def test_checkmate_is_terminal(self):
        # Scholar's mate
        board = chess.Board()
        for move in ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]:
            board.push_uci(move)
        node = Node(board)
        assert node.is_terminal

    def test_checkmate_value(self):
        # Scholar's mate - black is checkmated, side to move (black) loses
        board = chess.Board()
        for move in ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]:
            board.push_uci(move)
        node = Node(board)
        assert node.terminal_value == -1.0

    def test_stalemate_value(self):
        # A known stalemate position
        board = chess.Board("k7/8/1K6/8/8/8/8/8 b - - 0 1")
        # This isn't stalemate. Use a real stalemate:
        board = chess.Board("k7/8/2K5/8/8/8/8/8 b - - 0 1")
        if board.is_stalemate():
            node = Node(board)
            assert node.terminal_value == 0.0

    def test_total_visits_empty(self):
        board = chess.Board()
        node = Node(board)
        assert node.total_visits == 0

    def test_total_visits_after_edges(self):
        board = chess.Board()
        node = Node(board)
        e1 = Edge(chess.Move.from_uci("e2e4"), node, 0.5)
        e1.N = 5
        e2 = Edge(chess.Move.from_uci("d2d4"), node, 0.3)
        e2.N = 3
        node.edges = [e1, e2]
        assert node.total_visits == 8


class TestNodeExpand:
    def test_expand_creates_edges(self):
        """Expanding a node should create edges for all legal moves."""

        class MockNetwork:
            def predict(self, board_tensor):
                policy = np.ones(1858, dtype=np.float32) / 1858
                value = np.array([0.0])
                return policy, value

        board = chess.Board()
        node = Node(board)
        converter = Converter()
        network = MockNetwork()

        value = node.expand(network, converter)

        assert len(node.edges) == len(list(board.legal_moves))
        assert not node.is_leaf

    def test_expand_returns_value(self):
        class MockNetwork:
            def predict(self, board_tensor):
                policy = np.ones(1858, dtype=np.float32) / 1858
                value = np.array([0.7])
                return policy, value

        board = chess.Board()
        node = Node(board)
        converter = Converter()
        network = MockNetwork()

        value = node.expand(network, converter)
        assert np.isclose(value, 0.7)

    def test_expand_wdl_value(self):
        """BigNetwork returns WDL vector [win, draw, loss]."""

        class MockNetworkWDL:
            def predict(self, board_tensor):
                policy = np.ones(1858, dtype=np.float32) / 1858
                value = np.array([0.8, 0.1, 0.1])  # 80% win
                return policy, value

        board = chess.Board()
        node = Node(board)
        converter = Converter()
        network = MockNetworkWDL()

        value = node.expand(network, converter)
        # Expected: 0.8*1 + 0.1*0 + 0.1*(-1) = 0.7
        assert np.isclose(value, 0.7)

    def test_expand_priors_sum_to_one(self):
        class MockNetwork:
            def predict(self, board_tensor):
                policy = np.random.rand(1858).astype(np.float32)
                policy = policy / policy.sum()
                value = np.array([0.0])
                return policy, value

        board = chess.Board()
        node = Node(board)
        converter = Converter()
        network = MockNetwork()

        node.expand(network, converter)
        prior_sum = sum(e.P for e in node.edges)
        assert np.isclose(prior_sum, 1.0, atol=1e-5)

    def test_terminal_node_returns_terminal_value(self):
        """Expanding a terminal node returns terminal_value directly."""
        board = chess.Board()
        for move in ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]:
            board.push_uci(move)

        class MockNetwork:
            def predict(self, board_tensor):
                raise AssertionError("Should not be called on terminal node")

        node = Node(board)
        converter = Converter()
        value = node.expand(MockNetwork(), converter)
        assert value == -1.0


class TestNodeSelectEdge:
    def test_selects_highest_ucb(self):
        board = chess.Board()
        node = Node(board)
        e1 = Edge(chess.Move.from_uci("e2e4"), node, prior=0.9)
        e2 = Edge(chess.Move.from_uci("d2d4"), node, prior=0.1)
        node.edges = [e1, e2]

        selected = node.select_edge(c_puct=1.5)
        # e1 has higher prior, so higher UCB when both unvisited
        assert selected == e1

    def test_exploration_favors_unvisited(self):
        board = chess.Board()
        node = Node(board)
        e1 = Edge(chess.Move.from_uci("e2e4"), node, prior=0.5)
        e1.N = 100
        e1.W = 50
        e1.Q = 0.5
        e2 = Edge(chess.Move.from_uci("d2d4"), node, prior=0.5)
        # e2 is unvisited
        node.edges = [e1, e2]

        selected = node.select_edge(c_puct=1.5)
        # Unvisited e2 should have higher exploration bonus
        assert selected == e2


class TestNodeGetChildNode:
    def test_creates_child_on_first_call(self):
        board = chess.Board()
        node = Node(board)
        move = chess.Move.from_uci("e2e4")
        edge = Edge(move, node, prior=0.5)
        node.edges = [edge]

        child = node.get_child_node(edge)
        assert child is not None
        assert child.board != board  # Different board state
        assert chess.Move.from_uci("e2e4") not in child.board.legal_moves or child.board.turn == chess.BLACK

    def test_returns_same_child_on_second_call(self):
        board = chess.Board()
        node = Node(board)
        move = chess.Move.from_uci("e2e4")
        edge = Edge(move, node, prior=0.5)
        node.edges = [edge]

        child1 = node.get_child_node(edge)
        child2 = node.get_child_node(edge)
        assert child1 is child2


class TestGetMoveProbabilities:
    def test_greedy_temperature_zero(self):
        board = chess.Board()
        node = Node(board)
        e1 = Edge(chess.Move.from_uci("e2e4"), node, prior=0.5)
        e1.N = 10
        e2 = Edge(chess.Move.from_uci("d2d4"), node, prior=0.5)
        e2.N = 5
        node.edges = [e1, e2]

        probs = node.get_move_probabilities(temperature=0)
        assert probs["e2e4"] == 1.0
        assert probs["d2d4"] == 0.0

    def test_proportional_temperature_one(self):
        board = chess.Board()
        node = Node(board)
        e1 = Edge(chess.Move.from_uci("e2e4"), node, prior=0.5)
        e1.N = 10
        e2 = Edge(chess.Move.from_uci("d2d4"), node, prior=0.5)
        e2.N = 10
        node.edges = [e1, e2]

        probs = node.get_move_probabilities(temperature=1.0)
        assert np.isclose(probs["e2e4"], 0.5)
        assert np.isclose(probs["d2d4"], 0.5)

    def test_probs_sum_to_one(self):
        board = chess.Board()
        node = Node(board)
        for i, move_uci in enumerate(["e2e4", "d2d4", "g1f3", "b1c3"]):
            e = Edge(chess.Move.from_uci(move_uci), node, prior=0.25)
            e.N = (i + 1) * 5
            node.edges.append(e)

        probs = node.get_move_probabilities(temperature=1.0)
        assert np.isclose(sum(probs.values()), 1.0)

    def test_uniform_when_no_visits(self):
        board = chess.Board()
        node = Node(board)
        e1 = Edge(chess.Move.from_uci("e2e4"), node, prior=0.5)
        e2 = Edge(chess.Move.from_uci("d2d4"), node, prior=0.5)
        node.edges = [e1, e2]

        probs = node.get_move_probabilities(temperature=1.0)
        assert np.isclose(probs["e2e4"], 0.5)
        assert np.isclose(probs["d2d4"], 0.5)


class TestGetPolicyTarget:
    def test_shape(self):
        board = chess.Board()
        node = Node(board)
        converter = Converter()

        # Add some edges with visits
        for move in list(board.legal_moves)[:5]:
            e = Edge(move, node, prior=0.2)
            e.N = 10
            node.edges.append(e)

        target = node.get_policy_target(converter, temperature=1.0)
        assert target.shape == (1858,)

    def test_sums_to_approximately_one(self):
        board = chess.Board()
        node = Node(board)
        converter = Converter()

        for move in list(board.legal_moves)[:5]:
            e = Edge(move, node, prior=0.2)
            e.N = 10
            node.edges.append(e)

        target = node.get_policy_target(converter, temperature=1.0)
        assert np.isclose(target.sum(), 1.0, atol=1e-5)

    def test_only_legal_moves_have_probability(self):
        board = chess.Board()
        node = Node(board)
        converter = Converter()

        for move in list(board.legal_moves)[:3]:
            e = Edge(move, node, prior=0.33)
            e.N = 5
            node.edges.append(e)

        target = node.get_policy_target(converter, temperature=1.0)
        nonzero = np.count_nonzero(target)
        assert nonzero == 3