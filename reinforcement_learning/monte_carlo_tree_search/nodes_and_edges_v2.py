"""Alpha-Zero like Reinforcement Learning with Monte Carlo Tree Search for Chess"""

import math
from typing import Any

import chess
import numpy as np

from reinforcement_learning.helpers.converter import Converter


class Edge:
    """Represents a move (action) connecting a parent node to a child node."""

    __slots__ = ["move", "parent_node", "child_node", "N", "W", "Q", "P"]

    def __init__(self, move: chess.Move, parent_node: "Node", prior: float):
        self.move = move
        self.parent_node = parent_node
        self.child_node = None  # Lazily assigned when the child is expanded
        self.N = 0  # Visit count
        self.W = 0.0  # Total accumulated value
        self.Q = 0.0  # Mean value (W / N)
        self.P = prior  # Prior probability from the network

    def update(self, value: float):
        """Backpropagate a value through this edge."""
        self.N += 1
        self.W += value
        self.Q = self.W / self.N

    def ucb_score(self, parent_visits: int, c_puct: float = 1.5) -> float:
        """Calculate the PUCT score for this edge.

        Args:
            parent_visits: Total visit count of the parent node
            c_puct: Exploration constant (higher = more exploration)

        Returns:
            PUCT score combining exploitation (Q) and exploration (U)
        """
        u = c_puct * self.P * math.sqrt(parent_visits) / (1 + self.N)
        return self.Q + u


class Node:
    """Represents a board position (state) in the search tree."""

    __slots__ = ["board", "parent_edge", "edges", "_is_terminal", "_terminal_value"]

    def __init__(self, board: chess.Board, parent_edge: Edge = None):
        self.board = board
        self.parent_edge = parent_edge
        self.edges: list[Edge] = []
        self._is_terminal = None
        self._terminal_value = None

    @property
    def is_leaf(self) -> bool:
        """A leaf node has not been expanded yet."""
        return len(self.edges) == 0

    @property
    def is_terminal(self) -> bool:
        """A terminal node is checkmate, stalemate, or draw."""
        if self._is_terminal is None:
            self._is_terminal = self.board.is_game_over()
        return self._is_terminal

    @property
    def terminal_value(self) -> float:
        """Return the value of a terminal position from the perspective of the side to move.

        Returns:
            -1 if the side to move is checkmated, 0 for draws
        """
        if self._terminal_value is None:
            result = self.board.result()
            if result == "1/2-1/2":
                self._terminal_value = 0.0
            elif (result == "1-0" and self.board.turn == chess.WHITE) or (
                result == "0-1" and self.board.turn == chess.BLACK
            ):
                self._terminal_value = 1.0
            else:
                self._terminal_value = -1.0
        return self._terminal_value

    @property
    def total_visits(self) -> int:
        """Total visits across all child edges."""
        return sum(edge.N for edge in self.edges)

    def expand(self, network: Any, converter: Converter) -> float:
        """Expand this node by creating edges for all legal moves.

        Uses the network to evaluate the position and assign prior
        probabilities to each legal move.

        Args:
            network: The chess neural network
            converter: Converter for board/move encoding

        Returns:
            The network's value estimate for this position
        """
        if self.is_terminal:
            return self.terminal_value

        # Get network prediction
        board_tensor = converter.board_to_input_tensor(self.board)
        policy, value = network.predict(board_tensor)

        # Build reverse lookup once per expansion
        index_lookup = converter.index_lookup

        # Create edges with prior probabilities for each legal move
        for move in self.board.legal_moves:
            lookup_key = move_to_lookup_key(move, self.board.turn)

            if lookup_key in index_lookup:
                prior = float(policy[index_lookup[lookup_key]])
            else:
                raise AttributeError(f"Lookup key {lookup_key} not found in lookup")
            edge = Edge(move, self, prior)
            self.edges.append(edge)

        # Normalize priors so they sum to 1 over legal moves
        prior_sum = sum(edge.P for edge in self.edges)
        if prior_sum > 0:
            for edge in self.edges:
                edge.P /= prior_sum

        # The network may return different value formats:
        # - a scalar (smaller network: shape (1,) or 0-d array)
        # - a WDL probability vector (big network: shape (3,) -> [win, draw, loss])
        # Convert to a single scalar in [-1, 1] where win=1, draw=0, loss=-1
        value = np.asarray(value).reshape(-1)  # normalize to 1-D

        if value.size == 3:
            # WDL vector: expected value in [-1, 1]
            return float(np.dot(value, [1.0, 0.0, -1.0]))
        elif value.size == 1:
            # Scalar output from smaller network
            return float(value[0])
        else:
            raise ValueError(f"Unexpected value shape: {value.shape}")

    def select_edge(self, c_puct: float = 1.5) -> Edge:
        """Select the edge with the highest PUCT score.

        Args:
            c_puct: Exploration constant

        Returns:
            The edge with the highest UCB score
        """
        parent_visits = self.total_visits
        best_edge = None
        best_score = float("-inf")

        for edge in self.edges:
            score = edge.ucb_score(parent_visits, c_puct)
            if score > best_score:
                best_score = score
                best_edge = edge

        return best_edge

    def get_child_node(self, edge: Edge) -> "Node":
        """Get or create the child node for a given edge.

        Uses board.copy() + push() instead of deepcopy for performance.

        Args:
            edge: The edge to follow

        Returns:
            The child Node
        """
        if edge.child_node is None:
            child_board = self.board.copy()
            child_board.push(edge.move)
            edge.child_node = Node(child_board, edge)
        return edge.child_node

    def get_move_probabilities(self, temperature: float = 1.0) -> dict[str, float]:
        """Get move probabilities based on visit counts.

        Used during self-play to select moves and generate training targets.

        Args:
            temperature: Controls exploration.
                0 = always pick most visited (greedy)
                1 = proportional to visit counts
                >1 = more uniform/exploratory

        Returns:
            Dict mapping UCI move strings to probabilities
        """
        if temperature == 0:
            # Greedy: all probability on the most visited move
            best_edge = max(self.edges, key=lambda e: e.N)
            return {
                edge.move.uci(): (1.0 if edge is best_edge else 0.0)
                for edge in self.edges
            }

        # Proportional to N^(1/temperature)
        visits = np.array([edge.N for edge in self.edges], dtype=np.float64)
        visits = np.power(visits, 1.0 / temperature)
        total = visits.sum()

        if total == 0:
            # No visits yet, uniform distribution
            uniform = 1.0 / len(self.edges)
            return {edge.move.uci(): uniform for edge in self.edges}

        probs = visits / total
        return {edge.move.uci(): float(probs[i]) for i, edge in enumerate(self.edges)}

    def get_policy_target(
        self, converter: Converter, temperature: float = 1.0
    ) -> np.ndarray:
        """Generate a policy training target from MCTS visit counts.

        Returns a full 1858-length vector suitable for training the network.

        Args:
            converter: Converter for move index lookup
            temperature: Controls how sharp the distribution is

        Returns:
            numpy array of shape (1858,) with visit-count-based probabilities
        """
        index_lookup = converter.index_lookup
        move_probs = self.get_move_probabilities(temperature)
        target = np.zeros(len(converter.lookup), dtype=np.float32)

        for move_uci, prob in move_probs.items():
            move = chess.Move.from_uci(move_uci)
            lookupkey = move_to_lookup_key(move, self.board.turn)
            if lookupkey in index_lookup:
                target[index_lookup[lookupkey]] = prob

        return target


def mirror_move_uci(move_uci: str) -> str:
    """Mirror a UCI move string vertically (flip ranks).

    e.g. 'e2e4' -> 'e7e5', 'a7a8n' -> 'a2a1n'
    """

    def flip_rank(c):
        return str(9 - int(c))

    result = move_uci[0] + flip_rank(move_uci[1]) + move_uci[2] + flip_rank(move_uci[3])
    if len(move_uci) > 4:
        result += move_uci[4]  # Promotion piece
    return result


def move_to_lookup_key(move: chess.Move, board_turn: chess.Color) -> str:
    """Convert a chess.Move to the key used in the move lookup.

    Handles mirroring for black's turn and queen promotion stripping.

    Args:
        move: The chess move
        board_turn: Whose turn it is on the real board

    Returns:
        The lookup key string
    """
    move_uci = move.uci()

    # Mirror to friendly perspective if black's turn
    if board_turn == chess.BLACK:
        move_uci = mirror_move_uci(move_uci)

    # Queen promotions use the base move (without 'q' suffix)
    if move.promotion == chess.QUEEN:
        move_uci = move_uci[:-1]

    return move_uci