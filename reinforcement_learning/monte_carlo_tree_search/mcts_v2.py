"""Monte Carlo Tree Search for AlphaZero-style chess self-play.

Includes:
  - Sequential MCTS (``search``) and batched virtual-loss MCTS (``search_batched``)
    that evaluates leaves in groups for efficient GPU use.
  - An evaluation cache shared across the whole search/game: transposed
    positions reuse (priors, value) instead of re-running the network — network
    calls are the dominant self-play cost.
  - First Play Urgency (FPU) so the tree searches deep instead of broad.
  - ``SelfPlayGame``: one self-play game with optional resignation adjudication
    and a PGN record of the moves played.
  - ``play_games_parallel``: generate several games across CPU processes.
"""

import multiprocessing as mp
import time
from typing import Any

import chess
import chess.pgn
import numpy as np

from helpers.converter import Converter
from monte_carlo_tree_search.nodes_and_edges_v2 import Node, Edge


class MCTS:
    """Monte Carlo Tree Search using a neural network for evaluation and priors.

    Args:
        network: The chess neural network for position evaluation
        converter: Converter for board/move encoding
        num_simulations: Number of MCTS simulations per move
        c_puct: Exploration constant for PUCT formula
        dirichlet_alpha: Alpha parameter for Dirichlet noise at the root
        dirichlet_epsilon: Weight of Dirichlet noise vs network prior
        fpu_reduction: First Play Urgency reduction. An unvisited edge's Q is
            taken as ``parent value_estimate - fpu_reduction`` (clamped to -1)
            instead of 0.0. Without this, in losing positions (all explored
            Q < 0) every untried move looks better than every explored one, so
            the tree grows broad instead of deep (the ~4-ply problem).
            Leela-typical range 0.2-0.4. Set to None to restore old behavior.
        eval_cache_max: Max entries in the evaluation cache (position ->
            (legal-move priors, value)). Entries are ~(#legal moves) floats, so
            even 200k entries is only tens of MB. FIFO eviction when full.
    """

    def __init__(
        self,
        network: Any,
        converter: Converter,
        num_simulations: int = 100,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        fpu_reduction: float | None = 0.3,
        eval_cache_max: int = 200_000,
    ):
        self.network = network
        self.converter = converter
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.fpu_reduction = fpu_reduction

        # Evaluation cache. IMPORTANT: entries are only valid for the weights
        # that produced them — call clear_cache() after (re)loading/training
        # weights if you keep using the same MCTS object.
        self.eval_cache_max = eval_cache_max
        self._eval_cache: dict = {}
        self.cache_hits = 0
        self.cache_misses = 0

    # ------------------------------------------------------------------
    # Evaluation cache
    # ------------------------------------------------------------------

    def clear_cache(self):
        """Drop all cached evaluations (call after the network weights change)."""
        self._eval_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    @staticmethod
    def _position_key(board: chess.Board):
        """Cache key for a position's network evaluation.

        Uses python-chess's transposition key (pieces, turn, castling, en
        passant) *plus the halfmove clock*: the clock must be included because
        it is part of the 20-plane input encoding (plane 17), so two positions
        differing only in the clock genuinely evaluate differently.
        """
        try:
            return (board._transposition_key(), board.halfmove_clock)
        except AttributeError:  # private API fallback: FEN minus move counters
            return (" ".join(board.fen().split(" ")[:4]), board.halfmove_clock)

    def _cache_put(self, key, priors: np.ndarray, value: float):
        cache = self._eval_cache
        if len(cache) >= self.eval_cache_max:
            cache.pop(next(iter(cache)))  # FIFO eviction (dicts keep insert order)
        cache[key] = (priors, value)

    def _expand_leaf(self, node: Node) -> float:
        """Evaluate + expand a single leaf, via the cache when possible.

        Returns the scalar value of the position from the node's perspective.
        This replaces direct Node.expand calls in the search paths.
        """
        if node.is_terminal:
            return node.terminal_value

        key = self._position_key(node.board)
        hit = self._eval_cache.get(key)
        if hit is not None:
            self.cache_hits += 1
            priors, value = hit
        else:
            self.cache_misses += 1
            tensor = self.converter.board_to_input_tensor(node.board)
            policy, raw_value = self.network.predict(tensor)
            moves, priors = node._compute_priors(policy, self.converter)
            value = Node._scalar_value(raw_value)
            self._cache_put(key, priors, value)
            node.value_estimate = value
            if node.is_leaf:
                node._create_edges_from_priors(priors, moves)
            return value

        node.value_estimate = value
        if node.is_leaf:
            node._create_edges_from_priors(priors)
        return value

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------

    def _fpu(self, node: Node, at_noised_root: bool) -> float | None:
        """First Play Urgency value for the children of ``node``.

        Everywhere: the parent's own value estimate minus a reduction — an
        unvisited move is assumed slightly worse than the position it comes
        from, so the search deepens promising lines instead of fanning out.

        At the root *while Dirichlet noise is active*: optimistic FPU (+1.0),
        the lc0 convention, so the injected noise can actually pull unvisited
        root moves into the search (a pessimistic root FPU would mute the
        exploration noise exists to provide).
        """
        if self.fpu_reduction is None:
            return None  # legacy behavior: unvisited Q = 0.0
        if at_noised_root:
            return 1.0
        v = node.value_estimate - self.fpu_reduction
        return -1.0 if v < -1.0 else v

    # ------------------------------------------------------------------
    # Sequential search
    # ------------------------------------------------------------------

    def search(self, root: Node, add_noise: bool = True) -> Node:
        """Run sequential MCTS simulations (one leaf eval per network call).

        Args:
            root: The root node to search from
            add_noise: Whether to add Dirichlet noise to root priors (True during self-play)

        Returns:
            The root node with updated visit counts
        """
        if root.is_terminal:
            return root
        # Expand root if it's a leaf
        if root.is_leaf:
            self._expand_leaf(root)

        # Add Dirichlet noise to root priors for exploration
        noise_on = add_noise and len(root.edges) > 0
        if noise_on:
            self._add_dirichlet_noise(root)

        for _ in range(self.num_simulations):
            node = root
            search_path: list[Edge] = []

            # 1. SELECT — walk down the tree using PUCT until we hit a leaf
            while not node.is_leaf and not node.is_terminal:
                fpu = self._fpu(node, noise_on and node is root)
                edge = node.select_edge(self.c_puct, fpu)
                search_path.append(edge)
                node = node.get_child_node(edge)

            # 2. EXPAND & EVALUATE — expand the leaf and get its value
            #    (terminal value, cached evaluation, or network call)
            value = self._expand_leaf(node)

            # 3. BACKPROPAGATE — update all edges in the search path
            # Value must be flipped at each level since players alternate
            self._backpropagate(search_path, value)

        return root

    def _backpropagate(self, search_path: list[Edge], value: float):
        """Backpropagate the value up the search path, flipping at each level.

        The value from the network is from the perspective of the node that was
        expanded. As we go up the tree, we flip the sign because what's good
        for one player is bad for the other.
        """
        for edge in reversed(search_path):
            # The value at the expanded node is from that node's perspective.
            # The edge connects parent -> child, so the parent wants the
            # negated value (opponent's loss is my gain).
            value = -value
            edge.update(value)

    # ------------------------------------------------------------------
    # Batched search
    # ------------------------------------------------------------------

    def search_batched(
        self, root: Node, add_noise: bool = False, batch_size: int = 16
    ) -> Node:
        """MCTS that batches leaf evaluations to use the GPU efficiently.

        Plain MCTS evaluates one leaf per network call (batch size 1), which
        leaves a deep ResNet's GPU mostly idle. Here we collect up to
        ``batch_size`` leaves per round using *virtual loss* (a temporary penalty
        on each selected path so concurrent selections diverge to different
        leaves), looks each leaf up in the evaluation cache, evaluates the
        misses in a single network call, then backs up.

        Equivalent in search quality to the sequential version for typical
        batch sizes, but far higher throughput on GPU.

        Args:
            root: Root node to search from.
            add_noise: Add Dirichlet noise to root priors (self-play exploration).
            batch_size: Max leaves gathered per network call.

        Returns:
            The root node with updated visit counts.
        """
        if root.is_terminal:
            return root
        if root.is_leaf:
            self._expand_leaf(root)
        noise_on = add_noise and len(root.edges) > 0
        if noise_on:
            self._add_dirichlet_noise(root)

        sims_done = 0
        while sims_done < self.num_simulations:
            n_this = min(batch_size, self.num_simulations - sims_done)

            leaves: list[Node] = []          # unique nodes needing an evaluation
            leaf_index: dict[int, int] = {}  # id(node) -> position in `leaves`
            pending: list[tuple[list[Edge], Node]] = []  # (path, leaf) per sim

            for _ in range(n_this):
                node = root
                path: list[Edge] = []

                # SELECT — walk down with virtual loss until a leaf/terminal
                while not node.is_leaf and not node.is_terminal:
                    fpu = self._fpu(node, noise_on and node is root)
                    edge = node.select_edge(self.c_puct, fpu)
                    edge.add_virtual_loss()
                    path.append(edge)
                    node = node.get_child_node(edge)

                if node.is_terminal:
                    # No network eval needed; back up the exact value now.
                    self._backpropagate_batched(path, node.terminal_value)
                    continue

                if id(node) not in leaf_index:
                    leaf_index[id(node)] = len(leaves)
                    leaves.append(node)
                pending.append((path, node))

            if leaves:
                # EVALUATE — cache lookups + one batched call for the misses,
                # EXPAND each distinct leaf once (done inside the helper)
                values = self._evaluate_and_expand_batch(leaves)

                # BACKUP — every collected sim, using its leaf's value
                for path, node in pending:
                    self._backpropagate_batched(path, values[leaf_index[id(node)]])

            sims_done += n_this

        return root

    def _evaluate_and_expand_batch(self, leaves: list[Node]) -> list[float]:
        """Evaluate distinct leaves (cache first, one network call for misses)
        and expand each; returns scalar values aligned with ``leaves``."""
        n = len(leaves)
        priors_list: list = [None] * n
        values: list[float] = [0.0] * n
        keys = [self._position_key(node.board) for node in leaves]

        miss_idx: list[int] = []
        for i, key in enumerate(keys):
            hit = self._eval_cache.get(key)
            if hit is not None:
                self.cache_hits += 1
                priors_list[i], values[i] = hit
            else:
                self.cache_misses += 1
                miss_idx.append(i)

        if miss_idx:
            tensors = [
                self.converter.board_to_input_tensor(leaves[i].board)
                for i in miss_idx
            ]
            policies, raw_values = self.network.predict_batch(tensors)
            for j, i in enumerate(miss_idx):
                node = leaves[i]
                moves, priors = node._compute_priors(policies[j], self.converter)
                value = Node._scalar_value(raw_values[j])
                values[i] = value
                self._cache_put(keys[i], priors, value)
                node.value_estimate = value
                if node.is_leaf:
                    node._create_edges_from_priors(priors, moves)

        # Cache hits: expand from stored priors (misses were expanded above)
        for i, node in enumerate(leaves):
            if priors_list[i] is None:
                continue
            node.value_estimate = values[i]
            if node.is_leaf:
                node._create_edges_from_priors(priors_list[i])

        return values

    def _backpropagate_batched(self, search_path: list[Edge], value: float):
        """Back up through a path that had virtual loss applied during selection.

        Same sign-flipping as _backpropagate, but uses
        revert_virtual_loss_and_update so the virtual loss added in the SELECT
        phase is undone (N stays, the -1 is removed and the real value folded in).
        """
        for edge in reversed(search_path):
            value = -value
            edge.revert_virtual_loss_and_update(value)

    # ------------------------------------------------------------------
    # Root noise / move selection / tree reuse
    # ------------------------------------------------------------------

    def _add_dirichlet_noise(self, root: Node):
        """Mix Dirichlet noise into the root priors for exploration.

        This ensures the search doesn't collapse to always picking the
        network's top move, which is important early in training when
        the network is essentially random.

        The mix is always computed from the *clean* network priors
        (``edge.P_orig``), never from the current ``edge.P``. Self-play calls
        the search on every move, and after ``reuse_subtree`` the new root is a
        node whose priors may already contain noise from a previous call —
        mixing on top of that compounded move after move. Recomputing from
        P_orig makes noise application idempotent: fresh noise per search
        (standard AlphaZero), zero accumulation.
        """
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(root.edges))
        eps = self.dirichlet_epsilon
        for i, edge in enumerate(root.edges):
            edge.P = (1 - eps) * edge.P_orig + eps * noise[i]

    def get_best_move(self, root: Node, temperature: float = 1.0) -> chess.Move:
        """Select a move from the root based on visit counts.

        Args:
            root: The root node after search
            temperature: Controls exploration vs exploitation
                0 = always pick the most visited move
                1 = sample proportional to visit counts

        Returns:
            The selected chess move
        """
        if temperature == 0:
            # Greedy — pick the most visited edge
            best_edge = max(root.edges, key=lambda e: e.N)
            return best_edge.move

        # Sample proportional to visit counts
        visits = np.array([edge.N for edge in root.edges], dtype=np.float64)
        visits = np.power(visits, 1.0 / temperature)
        probs = visits / visits.sum()
        idx = np.random.choice(len(root.edges), p=probs)
        return root.edges[idx].move

    def advance_root(self, prev_root: Node | None, board: chess.Board) -> Node:
        """Return a search root for ``board``, reusing the previous search tree
        when ``board`` is a continuation of it (subtree reuse for the
        inference path — self-play already reuses via SelfPlayGame).

        If ``prev_root`` is None, belongs to a different game, or is more than
        a few plies behind, a fresh root is built instead. A final FEN equality
        check guards correctness, so callers can pass anything.
        """
        if prev_root is not None:
            prev_stack = prev_root.board.move_stack
            new_stack = board.move_stack
            gap = len(new_stack) - len(prev_stack)
            if 0 <= gap <= 4 and new_stack[: len(prev_stack)] == prev_stack:
                node = prev_root
                try:
                    for move in new_stack[len(prev_stack):]:
                        node = self.reuse_subtree(node, move)
                except (ValueError, AssertionError):
                    node = None
                if node is not None and node.board.fen() == board.fen():
                    return node
        return Node(board.copy())

    def reuse_subtree(self, root: Node, move: chess.Move) -> Node:
        """Reuse the subtree rooted at the child corresponding to the given move.

        Avoids rebuilding the entire tree from scratch after each move.

        Args:
            root: The current root node
            move: The move that was played

        Returns:
            The new root node (reused child if found, otherwise a fresh node)
        """
        for edge in root.edges:
            if edge.move == move:
                child = root.get_child_node(edge)
                child.parent_edge = None  # Detach from old tree for GC
                return child

        # Move not found in tree — create a fresh node
        new_board = root.board.copy()
        new_board.push(move)
        return Node(new_board)


class SelfPlayGame:
    """Plays a single self-play game using MCTS, collecting training data.

    Args:
        mcts: The MCTS instance to use for search
        temperature_threshold: Move number after which temperature drops to 0
        max_moves: Maximum number of moves before declaring a draw
        resign_threshold: Resign if the best edge's Q stays below this for
            ``resign_moves`` consecutive moves. None disables resignation.
        resign_moves: Consecutive below-threshold moves required to resign.
        resign_playout: If True, never actually resign but still track when the
            resignation condition WOULD have fired (see record fields). Used for
            the AlphaZero-style ~10% playout games that (a) measure the
            false-positive rate of the threshold and (b) keep won endgames in
            the training data so the net still learns to deliver mate.
        search_batch_size: Leaves per batched network call during search. 1 is
            effectively sequential (fine on CPU); higher helps on GPU.
    """

    def __init__(
        self,
        mcts: MCTS,
        temperature_threshold: int = 30,
        max_moves: int = 150,
        resign_threshold: float | None = None,
        resign_moves: int = 4,
        resign_playout: bool = False,
        search_batch_size: int = 8,
    ):
        self.mcts = mcts
        self.temperature_threshold = temperature_threshold
        self.max_moves = max_moves
        self.resign_threshold = resign_threshold
        self.resign_moves = resign_moves
        self.resign_playout = resign_playout
        self.search_batch_size = search_batch_size
        self.record = None  # populated by play(): result, move count, PGN, etc.

    def play(self) -> list[dict]:
        """Play a full self-play game and return training data.

        Returns:
            List of dicts, each containing:
                - board_tensor: numpy array (8, 8, 20)
                - policy_target: numpy array (1858,)
                - value_target: float (-1, 0, or 1) from the perspective of the side to move
        """
        board = chess.Board()
        root = Node(board)
        training_data = []
        move_count = 0
        resign_count = 0
        resigned_side = None
        would_resign_side = None  # first side that met the resign condition
        total_search_seconds = 0.0

        while (
            not board.is_game_over()
            and not self._claimable_draw(board)
            and move_count < self.max_moves
        ):
            # Pick temperature based on move number
            temperature = 1.0 if move_count < self.temperature_threshold else 0.0

            # Run MCTS (batched leaf evaluation), timing the search per move
            t_move = time.perf_counter()
            root = self.mcts.search_batched(
                root, add_noise=True, batch_size=self.search_batch_size
            )
            total_search_seconds += time.perf_counter() - t_move

            # --- Adjudication: resign if the side to move is hopelessly lost ---
            if self.resign_threshold is not None and root.edges:
                best_edge = max(root.edges, key=lambda e: e.N)
                if best_edge.Q < self.resign_threshold:
                    resign_count += 1
                else:
                    resign_count = 0
                if resign_count >= self.resign_moves and would_resign_side is None:
                    would_resign_side = board.turn  # condition fired for this side
                    if not self.resign_playout:
                        resigned_side = board.turn  # give up; opponent wins
                        break
                    # playout game: keep playing, we only wanted the measurement
            # ------------------------------------------------------------------

            # Store training sample (before making the move)
            board_tensor = self.mcts.converter.board_to_input_tensor(board)
            policy_target = root.get_policy_target(self.mcts.converter, temperature=1.0)
            training_data.append(
                {
                    "board_tensor": board_tensor,
                    "policy_target": policy_target,
                    "side_to_move": board.turn,
                }
            )

            # Select and play the move
            move = self.mcts.get_best_move(root, temperature=temperature)
            board.push(move)
            move_count += 1

            # Reuse the subtree for the next position
            root = self.mcts.reuse_subtree(root, move)

            if move_count % 20 == 0:
                print(f"  Move {move_count}: {board.fen()}")

        # Determine game outcome
        result = self._get_game_result(board, move_count, resigned_side)
        print(f"  Game over after {move_count} moves: {result}")

        # Build a PGN record of the game for logging (board holds the move stack)
        game_node = chess.pgn.Game.from_board(board)
        game_node.headers["Event"] = "self-play"
        game_node.headers["Result"] = result
        self.record = {
            "result": result,
            "num_moves": move_count,
            "num_positions": len(training_data),
            "resigned": resigned_side is not None,
            "resign_playout": self.resign_playout,
            # Which side met the resign condition (also tracked in playout games,
            # where it did NOT end the game). A playout game where this is set
            # but that side did not lose = a false positive of the threshold.
            "would_resign_side": (
                None if would_resign_side is None
                else ("white" if would_resign_side == chess.WHITE else "black")
            ),
            # Average wall time one MCTS move decision took in this game
            "avg_move_seconds": (
                round(total_search_seconds / move_count, 3) if move_count else None
            ),
            "pgn": str(game_node),
        }

        # Assign value targets based on game outcome
        return self._assign_values(training_data, result)

    @staticmethod
    def _claimable_draw(board: chess.Board) -> bool:
        """Claimable (not auto-applied) draw: 50-move rule or threefold repetition.

        Must mirror Node.is_terminal's draw conditions: the search scores these
        positions as terminal draws, so if one is actually reached in the game,
        the game has to end here — otherwise the next root is a terminal node
        with no edges and there is no policy target to extract. The engine
        *will* steer into repetitions when it is losing (a draw beats a loss),
        so this path is exercised in practice.
        """
        clock = board.halfmove_clock
        return clock >= 100 or (clock >= 8 and board.is_repetition(3))

    def _get_game_result(
        self, board: chess.Board, move_count: int, resigned_side=None
    ) -> str:
        """Get the game result string."""
        if resigned_side is not None:
            return "0-1" if resigned_side == chess.WHITE else "1-0"
        if board.is_game_over():
            return board.result()
        # Claimable draw (threefold / 50-move) or the max_moves cap
        return "1/2-1/2"

    def _assign_values(self, training_data: list[dict], result: str) -> list[dict]:
        """Assign value targets to each position based on the game outcome.

        The value is from the perspective of the side to move at each position.

        Each sample also gets a ``moves_left_target``: plies remaining until the
        end of the game as played (position i of an N-move game has N - i plies
        left; the final stored position has 1). Stored NOW, before the moves-left
        head exists, so the replay buffer is already fully labeled the day the
        head is added. For resigned / move-capped games this is the truncated
        game length — the same adjudication noise lc0 trains through.
        """
        if result == "1-0":
            white_value = 1.0
        elif result == "0-1":
            white_value = -1.0
        else:
            white_value = 0.0

        n = len(training_data)
        for i, sample in enumerate(training_data):
            if sample["side_to_move"] == chess.WHITE:
                sample["value_target"] = white_value
            else:
                sample["value_target"] = -white_value
            sample["moves_left_target"] = float(n - i)
            del sample["side_to_move"]  # No longer needed

        return training_data


# ----------------------------------------------------------------------
# Parallel self-play across CPU processes
# ----------------------------------------------------------------------

def _selfplay_worker(args):
    # Limit each worker to a single TF thread BEFORE importing TensorFlow, so N
    # workers use N cores cleanly instead of each grabbing all of them.
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # CPU-only workers

    (weight_path, lookup_path, num_simulations, num_res_blocks, num_filters,
     n_games, max_moves, temperature_threshold, resign_threshold,
     resign_playout_fraction, search_batch_size, seed) = args

    np.random.seed(seed) # distinct seed per worker, else identical games

    # Lazy import: keep TensorFlow out of the parent until a worker needs it.
    from networks.big_network import BigNetwork
    network = BigNetwork(
        num_res_blocks=num_res_blocks,
        num_filters=num_filters,
        lookup_path=lookup_path,
    )
    if weight_path:
        network.load(weight_path)

    converter = Converter(lookup_path=lookup_path)
    # One MCTS per worker: its evaluation cache stays valid for the whole batch
    # of games because the weights are fixed within this worker's lifetime.
    mcts = MCTS(network=network, converter=converter, num_simulations=num_simulations)

    games, records = [], []
    for _ in range(n_games):
        # AlphaZero-style: a fraction of games keeps the threshold for
        # measurement but plays to the end (false-positive rate + mating practice)
        playout = (
            resign_threshold is not None
            and np.random.rand() < resign_playout_fraction
        )
        game = SelfPlayGame(
            mcts,
            temperature_threshold=temperature_threshold,
            max_moves=max_moves,
            resign_threshold=resign_threshold,
            resign_playout=playout,
            search_batch_size=search_batch_size,
        )
        games.append(game.play())
        records.append(game.record)
    return games, records


def play_games_parallel(
    n_games: int,
    weight_path: str,
    lookup_path: str,
    num_simulations: int,
    num_res_blocks: int = 20,
    num_filters: int = 256,
    num_workers: int = None,
    max_moves: int = 150,
    temperature_threshold: int = 30,
    resign_threshold: float | None = None,
    resign_playout_fraction: float = 0.1,
    search_batch_size: int = 8,
):
    """Generate n_games of self-play across processes.

    Returns:
        (games, records)
          games   : list of games, each a list of position-sample dicts
                    (flatten before training)
          records : matching list of PGN record dicts (one per game)
    """
    if num_workers is None:
        num_workers = mp.cpu_count()

    # Split games across workers (each plays several to amortize TF startup)
    per_worker = [n_games // num_workers] * num_workers
    for i in range(n_games % num_workers):
        per_worker[i] += 1
    per_worker = [p for p in per_worker if p > 0]

    seeds = np.random.randint(0, 2**31 - 1, size=len(per_worker))
    args = [
        (weight_path, lookup_path, num_simulations, num_res_blocks, num_filters,
         g, max_moves, temperature_threshold, resign_threshold,
         resign_playout_fraction, search_batch_size, int(seeds[i]))
        for i, g in enumerate(per_worker)
    ]

    ctx = mp.get_context("spawn")  # spawn is the safe context for TensorFlow
    with ctx.Pool(processes=len(args)) as pool:
        results = pool.map(_selfplay_worker, args)

    all_games = [game for worker_games, _ in results for game in worker_games]
    all_records = [rec for _, worker_records in results for rec in worker_records]
    return all_games, all_records