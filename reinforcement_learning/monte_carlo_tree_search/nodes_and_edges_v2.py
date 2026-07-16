"""Alpha-Zero like Reinforcement Learning with Monte Carlo Tree Search for Chess"""

import math
from typing import Any

import chess
import numpy as np

from helpers.converter import Converter


class Edge:
    """Represents a move (action) connecting a parent node to a child node."""

    __slots__ = ["move", "parent_node", "child_node", "N", "W", "Q", "P", "P_orig"]

    def __init__(self, move: chess.Move, parent_node: "Node", prior: float):
        self.move = move
        self.parent_node = parent_node
        self.child_node = None  # Lazily assigned when the child is expanded
        self.N = 0  # Visit count
        self.W = 0.0  # Total accumulated value
        self.Q = 0.0  # Mean value (W / N)
        self.P = prior  # Effective prior (may include Dirichlet noise at the root)
        self.P_orig = prior  # Clean network prior — noise is always re-mixed from
        #                      this, so repeated noise application cannot compound.

    def update(self, value: float):
        """Backpropagate a value through this edge."""
        self.N += 1
        self.W += value
        self.Q = self.W / self.N
        self.parent_node._visit_total += 1  # keep the parent's cached total in sync

    def add_virtual_loss(self):
        """Temporarily discourage this edge during batched selection.

        Counts a pretend visit that lost (from this node's perspective), so other
        selections in the same batch diverge to different leaves. Reverted in
        revert_virtual_loss_and_update once the real value arrives.
        """
        self.N += 1
        self.W -= 1.0
        self.Q = self.W / self.N
        self.parent_node._visit_total += 1  # N changed here, so the total changes here

    def revert_virtual_loss_and_update(self, value: float):
        """Undo one virtual loss and fold in the real value.

        N was already incremented by add_virtual_loss during selection (and so was
        the parent's cached visit total), so we only adjust W (remove the -1
        virtual loss, add the real value) and recompute Q. Net effect equals one
        normal update(value).
        """
        self.W += 1.0 + value
        self.Q = self.W / self.N

    def ucb_score(
        self, parent_visits: int, c_puct: float = 1.5, fpu: float | None = None
    ) -> float:
        """Calculate the PUCT score for this edge.

        Args:
            parent_visits: Total visit count of the parent node
            c_puct: Exploration constant (higher = more exploration)
            fpu: First Play Urgency — the Q value assumed for an *unvisited* edge.
                None preserves the legacy behavior (unvisited Q = 0.0), which makes
                untried moves look better than every explored move in lost
                positions and drives the tree broad instead of deep.

        Returns:
            PUCT score combining exploitation (Q) and exploration (U)
        """
        q = self.Q if (self.N > 0 or fpu is None) else fpu
        u = c_puct * self.P * math.sqrt(parent_visits) / (1 + self.N)
        return q + u


class Node:
    """Represents a board position (state) in the search tree."""

    __slots__ = [
        "board",
        "parent_edge",
        "edges",
        "value_estimate",
        "_visit_total",
        "_is_terminal",
        "_terminal_value",
    ]

    def __init__(self, board: chess.Board, parent_edge: Edge = None):
        self.board = board
        self.parent_edge = parent_edge
        self.edges: list[Edge] = []
        self.value_estimate = 0.0  # network value from this node's perspective,
        #                            set on expansion; used for FPU of its children
        self._visit_total = 0  # running sum of edge.N — see total_visits
        self._is_terminal = None
        self._terminal_value = None

    @property
    def is_leaf(self) -> bool:
        """A leaf node has not been expanded yet."""
        return len(self.edges) == 0

    @property
    def is_terminal(self) -> bool:
        """A terminal node is checkmate, stalemate, or a draw.

        In addition to ``board.is_game_over()`` (checkmate, stalemate,
        insufficient material, 75-move, fivefold), the search treats these as
        terminal draws:

        - **claimable 50-move draws** (halfmove clock >= 100), and
        - **threefold repetition** (``board.is_repetition(3)``, which counts
          occurrences in the board's move stack — child boards are created with
          ``board.copy()``, so they carry the full game + search-path history).

        Neither is auto-claimed by python-chess, and the 20-plane input has no
        history, so without this the search shuffles pieces in won positions
        straight into repetition draws it cannot see coming.
        """
        if self._is_terminal is None:
            board = self.board
            clock = board.halfmove_clock
            if board.is_game_over():
                self._is_terminal = True
            # is_repetition(3) scans the move stack (expensive), so gate it:
            # a third occurrence needs >= 2 repetition cycles of >= 4 reversible
            # plies each, i.e. it is impossible while the halfmove clock is < 8.
            elif clock >= 100 or (clock >= 8 and board.is_repetition(3)):
                # Claimable draw: score it as a draw *now*, before it happens.
                self._is_terminal = True
                self._terminal_value = 0.0
            else:
                self._is_terminal = False
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
        """Total visits across all child edges.

        Maintained as a running counter (updated by Edge.update /
        Edge.add_virtual_loss) instead of summed over all edges on every
        selection — the old O(#edges) sum ran once per select_edge call and was
        measurable pure-Python overhead.
        """
        return self._visit_total

    def expand(self, network: Any, converter: Converter) -> float:
        """Expand this node by creating edges for all legal moves.

        Evaluates the position with the network (single inference), assigns prior
        probabilities to each legal move, and returns the scalar value estimate.

        Note: MCTS routes expansions through its evaluation cache
        (MCTS._expand_leaf) instead of calling this directly; this method remains
        for compatibility and for use without an MCTS instance.

        Args:
            network: The chess neural network
            converter: Converter for board/move encoding

        Returns:
            The network's value estimate for this position
        """
        if self.is_terminal:
            return self.terminal_value

        board_tensor = converter.board_to_input_tensor(self.board)
        policy, value = network.predict(board_tensor)
        self._create_edges(policy, converter)
        self.value_estimate = self._scalar_value(value)
        return self.value_estimate

    def expand_from_eval(
        self, policy: np.ndarray, converter: Converter, value: float | None = None
    ) -> None:
        """Expand using an already-computed policy (batched-search path).

        The network value is backed up separately by the caller (see
        MCTS.search_batched); pass it here as well so it can be stored as this
        node's value_estimate (needed for FPU of its children).
        """
        self._create_edges(policy, converter)
        if value is not None:
            self.value_estimate = value

    def _create_edges(self, policy: np.ndarray, converter: Converter) -> None:
        """Create one edge per legal move, with priors from a softmax over the
        legal moves' policy logits. See _compute_priors."""
        moves, priors = self._compute_priors(policy, converter)
        self._create_edges_from_priors(priors, moves)

    def _compute_priors(
        self, policy: np.ndarray, converter: Converter
    ) -> tuple[list[chess.Move], np.ndarray]:
        """Softmax the legal moves' raw policy logits into priors.

        The policy head emits RAW logits, so a move's prior is its softmax value
        restricted to the legal moves (equivalent to masking illegal logits to
        -inf and softmaxing). The priors therefore sum to 1 over legal moves and
        are ordered like ``list(self.board.legal_moves)``.

        Split out from edge creation so priors can be stored in / restored from
        the MCTS evaluation cache: for the same position, python-chess generates
        legal moves in the same order, so a cached prior vector re-aligns with a
        fresh ``list(board.legal_moves)``. Returns ``(moves, priors)`` so callers
        can pass the moves straight to _create_edges_from_priors without
        generating them a second time.
        """
        # Reverse lookup is cached on the converter (built once, not per expansion)
        index_lookup = converter.index_lookup

        moves = list(self.board.legal_moves)
        if not moves:
            return moves, np.empty(0, dtype=np.float64)

        logits = np.empty(len(moves), dtype=np.float64)
        turn = self.board.turn
        for i, move in enumerate(moves):
            lookup_key = move_to_lookup_key(move, turn)
            if lookup_key not in index_lookup:
                raise AttributeError(f"Lookup key {lookup_key} not found in lookup")
            logits[i] = policy[index_lookup[lookup_key]]

        # Numerically stable softmax over the legal-move logits.
        logits -= logits.max()
        priors = np.exp(logits)
        priors /= priors.sum()
        return moves, priors

    def _create_edges_from_priors(
        self, priors: np.ndarray, moves: list[chess.Move] | None = None
    ) -> None:
        """Build edges from a prior vector aligned with
        ``list(self.board.legal_moves)`` (see _compute_priors).

        Pass ``moves`` when the caller already generated the legal moves (the
        network-miss path) to skip a second generation; leave it None on the
        cache-hit path, where moves are regenerated in the same order.
        """
        if moves is None:
            moves = self.board.legal_moves
        edges = self.edges
        for move, prior in zip(moves, priors):
            edges.append(Edge(move, self, float(prior)))

    @staticmethod
    def _scalar_value(value) -> float:
        """Collapse the network value head to a scalar in [-1, 1].

        The network may return different value formats:
          - a WDL probability vector (big network: shape (3,) -> win/draw/loss)
          - a scalar (smaller network: shape (1,) or 0-d)
        Mapped as win=1, draw=0, loss=-1.
        """
        value = np.asarray(value).reshape(-1)  # normalize to 1-D
        if value.size == 3:
            return float(np.dot(value, [1.0, 0.0, -1.0]))
        elif value.size == 1:
            return float(value[0])
        else:
            raise ValueError(f"Unexpected value shape: {value.shape}")

    def select_edge(self, c_puct: float = 1.5, fpu: float | None = None) -> Edge:
        """Select the edge with the highest PUCT score.

        Args:
            c_puct: Exploration constant
            fpu: First Play Urgency — Q assumed for unvisited edges (None keeps
                the legacy unvisited-Q=0 behavior). Computed by the caller,
                typically ``parent value_estimate - fpu_reduction`` (see
                MCTS._fpu), so that in losing positions untried moves stop
                looking better than every explored move.

        Returns:
            The edge with the highest UCB score
        """
        # Inlined PUCT loop: hoist the sqrt and the c_puct multiply out of the
        # per-edge score (previously recomputed for every edge via ucb_score).
        exploration = c_puct * math.sqrt(self._visit_total)
        use_fpu = fpu is not None

        best_edge = None
        best_score = float("-inf")
        for edge in self.edges:
            n = edge.N
            q = fpu if (use_fpu and n == 0) else edge.Q
            score = q + exploration * edge.P / (1 + n)
            if score > best_score:
                best_score = score
                best_edge = edge

        return best_edge

    def get_child_node(self, edge: Edge) -> "Node":
        """Get or create the child node for a given edge.

        Uses board.copy() + push() instead of deepcopy for performance. The copy
        keeps the move stack (python-chess default), which is what makes
        is_repetition(3) work in is_terminal.

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