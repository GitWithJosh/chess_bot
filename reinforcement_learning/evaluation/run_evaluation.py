import chess
import chess.svg

from reinforcement_learning.evaluation.random_move_opponent import RandomMover
from reinforcement_learning.evaluation.evaluation_against_other_engine import EvaluationAgainstOtherEngine
from smaller_network import SmallerNetwork
from converter import Converter
from reinforcement_learning.evaluation.stockfish_opponent import StockfishOpponent

NETWORK = "small" # or later "big"
NETWORK_ITERATION = "v1"
NETWORK_WEIGTHS = "/home/timle/semester_6/chess_bot/reinforcement_learning/random_model.weights.h5"

OPPONENT_TYPE = "random" # or "stockfish"
AMOUNT_OF_GAMES = 10
SAVE_TO_PATH = f"{NETWORK}_network_{NETWORK_ITERATION}_vs_{OPPONENT_TYPE}_{AMOUNT_OF_GAMES}_games.pgn"


# Load Network
network = SmallerNetwork() if NETWORK == "small" else None
network.load(NETWORK_WEIGTHS)

# Get opponent
engine_opponent = StockfishOpponent() if OPPONENT_TYPE == "stockfish" else RandomMover()

converter = Converter()

# Run evaluation
evaluation = EvaluationAgainstOtherEngine(AMOUNT_OF_GAMES, network, engine_opponent, converter)

evaluation.play_games()
evaluation.save_pgn(SAVE_TO_PATH)
