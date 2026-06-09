import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# now simple sibling imports like `from helpers.converter import Converter` will work

from random_move_opponent import RandomMover
from evaluation_against_other_engine import (
    EvaluationAgainstOtherEngine,
)
from networks.smaller_network import SmallerNetwork
from networks.big_network import BigNetwork
from helpers.converter import Converter
from helpers.weights_utils import (
    get_latest_weights,
    extract_iteration,
)
from stockfish_opponent import StockfishOpponent

NETWORK = "small"  # or later "big"
NETWORK_ITERATION = "latest"  # or latest or "1" or "2" etc.
BASE_DIR = os.path.dirname(__file__)[: -len("evaluation")]
WEIGHTS_DIR = os.path.join(BASE_DIR, "networks", "weights")
OPPONENT_TYPE = "random"  # or "stockfish"
AMOUNT_OF_GAMES = 1

# Resolve weights path
if NETWORK_ITERATION == "latest":
    weights_path = get_latest_weights(NETWORK, WEIGHTS_DIR)
    NETWORK_ITERATION = extract_iteration(weights_path)
else:
    expected_name = f"{NETWORK}_network_v{NETWORK_ITERATION}.weights.h5"
    weights_path = os.path.join(WEIGHTS_DIR, expected_name)

if not os.path.isfile(weights_path):
    raise FileNotFoundError(f"Weights file not found: {weights_path}")

THIS_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(THIS_DIR, "results")
if not os.path.isdir(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
SAVE_TO_PATH = os.path.join(
    RESULTS_DIR,
    f"{NETWORK}_network_iteration_{NETWORK_ITERATION}_vs_{OPPONENT_TYPE}_{AMOUNT_OF_GAMES}_games.pgn",
)


# Load Network
network = SmallerNetwork() if NETWORK == "small" else BigNetwork()
network.load(weights_path)

# Get opponent
engine_opponent = StockfishOpponent() if OPPONENT_TYPE == "stockfish" else RandomMover()

converter = Converter()

# Run evaluation
evaluation = EvaluationAgainstOtherEngine(
    AMOUNT_OF_GAMES, network, engine_opponent, converter
)

evaluation.play_games()
evaluation.save_pgn(SAVE_TO_PATH)
