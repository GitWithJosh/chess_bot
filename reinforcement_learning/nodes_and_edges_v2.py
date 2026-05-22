"""Alpha-Zero like Reinforcement Learning with Monte Carlo Tree Search for Chess"""

import math
import chess
import numpy as np

from converter import Converter
from smaller_network import SmallerNetwork


class Edge:
    """Represents a move (action) connecting a parent node to a child node."""

    __slots__ = ["move", "parent_node", "child_node", "N", "W", "Q", "P"]

    def __init__(self, move: chess.Move, parent_node: "Node", prior: float):
        self.move = move
        self.parent_node = parent_node
        self.child_node = None  # Lazily assigned when the child is expanded
        self.N = 0      # Visit count
        self.W = 0.0    # Total accumulated value
        self.Q = 0.0    # Mean value (W / N)
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
            elif (result == "1-0" and self.board.turn == chess.WHITE) or \
                 (result == "0-1" and self.board.turn == chess.BLACK):
                self._terminal_value = 1.0
            else:
                self._terminal_value = -1.0
        return self._terminal_value

    @property
    def total_visits(self) -> int:
        """Total visits across all child edges."""
        return sum(edge.N for edge in self.edges)

    def expand(self, network: SmallerNetwork, converter: Converter) -> float:
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
        index_lookup = {v: int(k) for k, v in converter.lookup.items()}

        # Create edges with prior probabilities for each legal move
        for move in self.board.legal_moves:
            move_uci = move.uci()
            if move_uci.endswith("q"):
                prior = float(policy[index_lookup[move_uci[:-1]]])
            elif move_uci in index_lookup:
                prior = float(policy[index_lookup[move_uci]])
            else:
                raise AttributeError(f"Move {move_uci} not found in lookup")
            edge = Edge(move, self, prior)
            self.edges.append(edge)

        # Normalize priors so they sum to 1 over legal moves
        prior_sum = sum(edge.P for edge in self.edges)
        if prior_sum > 0:
            for edge in self.edges:
                edge.P /= prior_sum

        return float(value)

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
            return {edge.move.uci(): (1.0 if edge is best_edge else 0.0) for edge in self.edges}

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

    def get_policy_target(self, converter: Converter, temperature: float = 1.0) -> np.ndarray:
        """Generate a policy training target from MCTS visit counts.

        Returns a full 1858-length vector suitable for training the network.

        Args:
            converter: Converter for move index lookup
            temperature: Controls how sharp the distribution is

        Returns:
            numpy array of shape (1858,) with visit-count-based probabilities
        """
        index_lookup = {v: int(k) for k, v in converter.lookup.items()}
        move_probs = self.get_move_probabilities(temperature)
        target = np.zeros(len(converter.lookup), dtype=np.float32)

        for move_uci, prob in move_probs.items():
            if move_uci in index_lookup:
                target[index_lookup[move_uci]] = prob

        return target