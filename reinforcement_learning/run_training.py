import os
import re

import numpy as np
from networks.big_network import BigNetwork
from networks.smaller_network import SmallerNetwork
from monte_carlo_tree_search.mcts_v2 import MCTS, SelfPlayGame
from helpers.converter import Converter

# Configuration
NETWORK = "small"  # or later "big"
NETWORK_ITERATION = "latest"  # or number of iteration, CAREFUL when setting number, overwrites next weights file

# Resolve weights directory relative to this script (reinforcement_learning/networks/weights)
BASE_DIR = os.path.dirname(__file__)
WEIGHTS_DIR = os.path.join(BASE_DIR, "networks", "weights")

if not os.path.isdir(WEIGHTS_DIR):
    raise FileNotFoundError(f"Weights directory not found: {WEIGHTS_DIR}")


# Extract iteration number and sort by it
def extract_iteration(file_name):
    """Extract iteration number from a weights filename.

    Handles inputs like:
    - small_network_v0.weights.h5
    - /full/path/to/small_network_v10.weights.h5

    Returns -1 when no iteration is found.
    """
    base = os.path.basename(file_name)
    m = re.search(r"v(\d+)", base)
    if m:
        return int(m.group(1))
    return -1


if NETWORK_ITERATION == "latest":
    files = os.listdir(WEIGHTS_DIR)
    files = [f for f in files if f.endswith(".weights.h5") and f.startswith(NETWORK)]
    # If no files match the network prefix, fall back to any weights file
    if not files:
        files = [f for f in os.listdir(WEIGHTS_DIR) if f.endswith(".weights.h5")]
    if not files:
        raise FileNotFoundError(f"No weights found in {WEIGHTS_DIR}.")

    latest_file = max(files, key=extract_iteration)
    weights_path = os.path.join(WEIGHTS_DIR, latest_file)
else:
    expected_name = f"{NETWORK}_network_v{NETWORK_ITERATION}.weights.h5"
    weights_path = os.path.join(WEIGHTS_DIR, expected_name)

if not os.path.isfile(weights_path):
    raise FileNotFoundError(f"Weights file not found: {weights_path}")

if __name__ == "__main__":
    network = SmallerNetwork() if NETWORK == "small" else BigNetwork()
    network.load(weights_path)
    print(f"Loaded {NETWORK} network with weights from {weights_path}")
    converter = Converter()
    mcts = MCTS(network, converter, num_simulations=2)
    print(
        f"Initialized MCTS network. {mcts.num_simulations} iterations per move, c_puct={mcts.c_puct}"
    )
    self_play_game = SelfPlayGame(mcts)
    print(
        f"Starting self-play game with temperature threshold {self_play_game.temperature_threshold} and max moves {self_play_game.max_moves}"
    )
    training_data = self_play_game.play()
    print(f"\nCollected {len(training_data)} training samples")
    print(f"  Board tensor shape: {training_data[0]['board_tensor'].shape}")
    print(f"  Policy target shape: {training_data[0]['policy_target'].shape}")
    print(f"  Policy target sum: {training_data[0]['policy_target'].sum():.4f}")
    print(f"  Value target: {training_data[0]['value_target']}")

    # Show how to train the network from this data
    print("\nTraining on collected data...")
    board_tensors = np.array([s["board_tensor"] for s in training_data])
    policy_targets = np.array([s["policy_target"] for s in training_data])
    value_targets = np.array(
        [[s["value_target"]] for s in training_data], dtype=np.float32
    )

    history = network.train(
        board_tensors,
        policy_targets,
        value_targets,
        epochs=1,
        batch_size=32,
        verbose=1,
    )
    print(f"Training loss: {history.history['loss'][0]:.4f}")
    # save the updated weights after training
    print(f"iteration before was {extract_iteration(weights_path)}")
    iteration_number = extract_iteration(weights_path) + 1
    print(f"iteration after is {iteration_number}")
    updated_weights_path = os.path.join(
        WEIGHTS_DIR, f"{NETWORK}_network_v{iteration_number}.weights.h5"
    )
    network.save(updated_weights_path)
    print(f"Saved updated weights to {updated_weights_path}")
