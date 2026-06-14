"""Monte Carlo Tree Search for AlphaZero-style chess self-play."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU-only self-play
os.environ["OMP_NUM_THREADS"] = "1"        # one TF thread per worker — avoids oversubscription
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # quiet the logs from each worker

import multiprocessing as mp

from typing import Any

import chess
import numpy as np

from reinforcement_learning.helpers.converter import Converter
from reinforcement_learning.monte_carlo_tree_search.nodes_and_edges_v2 import Node, Edge


class MCTS:
    """Monte Carlo Tree Search using a neural network for evaluation and priors.

    Args:
        network: The chess neural network for position evaluation
        converter: Converter for board/move encoding
        num_simulations: Number of MCTS simulations per move
        c_puct: Exploration constant for PUCT formula
        dirichlet_alpha: Alpha parameter for Dirichlet noise at the root
        dirichlet_epsilon: Weight of Dirichlet noise vs network prior
    """

    def __init__(
        self,
        network: Any,
        converter: Converter,
        max_moves: int = 150,
        num_simulations: int = 100,

        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
    ):
        self.network = network
        self.converter = converter
        self.max_moves = max_moves
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def search(self, root: Node, add_noise: bool = True) -> Node:
        """Run MCTS simulations from the given root node.

        Args:
            root: The root node to search from
            add_noise: Whether to add Dirichlet noise to root priors (True during self-play)

        Returns:
            The root node with updated visit counts
        """
        # Expand root if it's a leaf
        if root.is_leaf:
            root.expand(self.network, self.converter)

        # Add Dirichlet noise to root priors for exploration
        if add_noise and len(root.edges) > 0:
            self._add_dirichlet_noise(root)

        for _ in range(self.num_simulations):
            node = root
            search_path: list[Edge] = []

            # 1. SELECT — walk down the tree using PUCT until we hit a leaf
            while not node.is_leaf and not node.is_terminal:
                edge = node.select_edge(self.c_puct)
                search_path.append(edge)
                node = node.get_child_node(edge)

            # 2. EXPAND & EVALUATE — expand the leaf and get the network value
            if node.is_terminal:
                value = node.terminal_value
            else:
                value = node.expand(self.network, self.converter)

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

    def _add_dirichlet_noise(self, root: Node):
        """Add Dirichlet noise to root node priors for exploration.

        This ensures the search doesn't collapse to always picking the
        network's top move, which is important early in training when
        the network is essentially random.
        """
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(root.edges))
        for i, edge in enumerate(root.edges):
            edge.P = (
                1 - self.dirichlet_epsilon
            ) * edge.P + self.dirichlet_epsilon * noise[i]

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
    """

    def __init__(
        self,
        mcts: MCTS,
        temperature_threshold: int = 30,
        max_moves: int = 300,
        resign_threshold: float = None, # Set None to disable, or ~-0.9
        resign_moves: int = 4,         # Consecutive moves below threshold before resigning
    ):
        self.mcts = mcts
        self.temperature_threshold = temperature_threshold
        self.max_moves = max_moves
        self.resign_threshold = resign_threshold
        self.resign_moves = resign_moves

    def play(self) -> list[dict]:
        """Play a full self-play game and return training data.

        Returns:
            List of dicts, each containing:
                - board_tensor: numpy array (8, 8, 112)
                - policy_target: numpy array (1858,)
                - value_target: float (-1, 0, or 1) from the perspective of the side to move
        """
        board = chess.Board()
        root = Node(board)
        training_data = []
        move_count = 0
        resign_count = 0
        resigned_side = None


        while not board.is_game_over() and move_count < self.max_moves:
            # Pick temperature based on move number
            temperature = 1.0 if move_count < self.temperature_threshold else 0.0

            # Run MCTS
            root = self.mcts.search(root, add_noise=True)

            # --- Adjudication: resign if the side to move is hopelessly lost ---
            if self.resign_threshold is not None and root.edges:
                best_edge = max(root.edges, key=lambda e: e.N)
                if best_edge.Q < self.resign_threshold:
                    resign_count += 1
                else:
                    resign_count = 0
                if resign_count >= self.resign_moves:
                    resigned_side = board.turn  # this side gives up; opponent wins
                    break
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

        # Assign value targets based on game outcome
        return self._assign_values(training_data, result)

    def _get_game_result(self, board: chess.Board, move_count: int, resigned_side=None) -> str:
        if resigned_side is not None:
            return "0-1" if resigned_side == chess.WHITE else "1-0"
        if move_count >= self.max_moves:
            return "1/2-1/2"
        return board.result()

    def _assign_values(self, training_data: list[dict], result: str) -> list[dict]:
        """Assign value targets to each position based on the game outcome.

        The value is from the perspective of the side to move at each position.
        """
        if result == "1-0":
            white_value = 1.0
        elif result == "0-1":
            white_value = -1.0
        else:
            white_value = 0.0

        for sample in training_data:
            if sample["side_to_move"] == chess.WHITE:
                sample["value_target"] = white_value
            else:
                sample["value_target"] = -white_value
            del sample["side_to_move"]  # No longer needed

        return training_data

def _selfplay_worker(args):
    (weight_path, lookup_path, num_simulations, num_res_blocks,
     num_filters, n_games, max_moves, temperature_threshold, seed) = args

    np.random.seed(seed)  # CRITICAL: without distinct seeds every worker plays identical games

    from networks.big_network import BigNetwork
    network = BigNetwork(num_res_blocks=num_res_blocks, num_filters=num_filters)
    if weight_path:
        network.load(weight_path)

    converter = Converter(lookup_path=lookup_path)
    mcts = MCTS(network=network, converter=converter, num_simulations=num_simulations)

    games = []
    for _ in range(n_games):
        game = SelfPlayGame(mcts, temperature_threshold=temperature_threshold,
                            max_moves=max_moves)
        games.append(game.play())
    return games

def play_games_parallel(
    n_games: int,
    weight_path: str,
    lookup_path: str,
    num_simulations: int,
    num_res_blocks: int = 10,
    num_filters: int = 256,
    num_workers: int = None,
    max_moves: int = 150,
    temperature_threshold: int = 30,
):
    """Generate n_games of self-play across processes.

    Returns a list of games, each a list of position-sample dicts
    (same nested shape as the old play_n_games — flatten before training).
    """
    if num_workers is None:
        num_workers = mp.cpu_count()

    # Split games across workers (each worker plays several to amortize TF startup)
    per_worker = [n_games // num_workers] * num_workers
    for i in range(n_games % num_workers):
        per_worker[i] += 1
    per_worker = [p for p in per_worker if p > 0]

    seeds = np.random.randint(0, 2**31 - 1, size=len(per_worker))
    args = [
        (weight_path, lookup_path, num_simulations, num_res_blocks,
         num_filters, g, max_moves, temperature_threshold, int(seeds[i]))
        for i, g in enumerate(per_worker)
    ]

    ctx = mp.get_context("spawn")  # spawn is the safe context for TensorFlow
    with ctx.Pool(processes=len(args)) as pool:
        results = pool.map(_selfplay_worker, args)

    return [game for worker_games in results for game in worker_games]


def play_n_games(n:int, load_network:bool, network_path:str, mcts:MCTS, converter:Converter, network):

    print(f"Initializing network and converter to play {n} games...")
    if load_network:
        network.load(network_path)

    training_data = []

    for i in range(n):
        game = SelfPlayGame(mcts, temperature_threshold=30, max_moves=300)
        game_data = game.play()
        training_data.append(game_data)

        print(f"Game {1} finished")

    return training_data

def train_on_game_batch(batch_size:int, weight_file):

    from networks.smaller_network import SmallerNetwork
    network = SmallerNetwork(num_res_blocks=5, num_filters=128)
    converter = Converter()
    mcts = MCTS(
        network=network,
        converter=converter,
        num_simulations=10
    )
    training_data = play_n_games(batch_size,True,weight_file,mcts,converter, network)

    print("\nTraining on collected data...")

    # TODO: DOES NOT WORK YET as training data is nested, list comprehension needs to b adjusted
    board_tensors = np.array([s["board_tensor"] for s in training_data])
    policy_targets = np.array([s["policy_target"] for s in training_data])
    value_targets = np.array([[s["value_target"]] for s in training_data], dtype=np.float32)

    history = network.train(
        board_tensors, policy_targets, value_targets,
        epochs=1, batch_size=32, verbose=1,
    )
    print(f"Training loss: {history.history['loss'][0]:.4f}")

def play_single_game():
    """Play one self-play game and print the results."""
    import time
    from reinforcement_learning.networks.big_network import BigNetwork

    print("Initializing network and converter...")
    network = BigNetwork(num_res_blocks=10, num_filters=256)
    converter = Converter()  # add lookup_path=... if not running from repo root

    mcts = MCTS(
        network=network,
        converter=converter,
        num_simulations=10,  # low for a speed test; use 100+ for real games
    )

    game = SelfPlayGame(
        mcts,
        temperature_threshold=30,
        max_moves=150,
        resign_threshold=None,  # explicitly disabled
    )

    print("Starting self-play game...")
    t0 = time.perf_counter()
    training_data = game.play()
    elapsed = time.perf_counter() - t0

    print(f"\nGame finished in {elapsed:.1f}s "
          f"({elapsed / max(len(training_data), 1):.2f}s per position)")
    print(f"Collected {len(training_data)} training samples")
    print(f"  Board tensor shape: {training_data[0]['board_tensor'].shape}")
    print(f"  Policy target sum:  {training_data[0]['policy_target'].sum():.4f}")
    print(f"  Value target:       {training_data[0]['value_target']}")

    # Show how to train the network from this data
    # print("\nTraining on collected data...")
    # board_tensors = np.array([s["board_tensor"] for s in training_data])
    # policy_targets = np.array([s["policy_target"] for s in training_data])
    # value_targets = np.array(
    #     [[s["value_target"]] for s in training_data], dtype=np.float32
    # )

    # history = network.train(
    #     board_tensors,
    #     policy_targets,
    #     value_targets,
    #     epochs=1,
    #     batch_size=32,
    #     verbose=1,
    # )
    # print(f"Training loss: {history.history['loss'][0]:.4f}")


if __name__ == "__main__":
    play_single_game()