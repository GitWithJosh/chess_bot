import os
import re

import numpy as np
from networks.big_network import BigNetwork
from networks.smaller_network import SmallerNetwork
from monte_carlo_tree_search.mcts_v2 import MCTS, SelfPlayGame
from helpers.converter import Converter
from helpers.weights_utils import get_latest_weights, extract_iteration

# Configuration
NETWORK = "big"  # or later "big"
NETWORK_ITERATION = "latest"  # or number of iteration, CAREFUL when setting number, overwrites next weights file

# Resolve weights directory relative to this script (reinforcement_learning/networks/weights)
BASE_DIR = os.path.dirname(__file__)
WEIGHTS_DIR = os.path.join(BASE_DIR, "networks", "weights")

if not os.path.isdir(WEIGHTS_DIR):
    raise FileNotFoundError(f"Weights directory not found: {WEIGHTS_DIR}")


# Resolve weights path (either latest or a specific iteration)
if NETWORK_ITERATION == "latest":
    weights_path = get_latest_weights(NETWORK, WEIGHTS_DIR)
else:
    expected_name = f"{NETWORK}_network_v{NETWORK_ITERATION}.weights.h5"
    weights_path = os.path.join(WEIGHTS_DIR, expected_name)

if not os.path.isfile(weights_path):
    raise FileNotFoundError(f"Weights file not found: {weights_path}")

if __name__ == "__main__":
    network = SmallerNetwork() if NETWORK == "small" else BigNetwork()
    print(f"Trying to load {NETWORK} network weights from {weights_path}...")
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
