import chess
import tensorflow as tf
from random_move_opponent import RandomMover
from converter import Converter
from nodes_and_edges_v2 import mirror_move_uci
from smaller_network import SmallerNetwork

class EvaluationAgainstOtherEngine:

    def __init__(self, amount_of_games:int, network:SmallerNetwork, other_engine, converter:Converter):
        self.amount_of_games = amount_of_games
        self.network = network
        self.converter = converter
        self.other_engine = other_engine
        self.results = {"network_wins": 0, "draws": 0, "network_losses": 0}


    def play_games(self):
        
        for i in range(self.amount_of_games):
            if i < self.amount_of_games // 2:
                starting_player = "network"
            else:
                starting_player = "other_engine"

            result = self._play_game(starting_player)

            if result == 1:
                self.results["network_wins"] += 1
            elif result == 0:
                self.results["network_losses"] += 1
            else:
                self.results["draws"] += 1

    def _play_game(self, starting_player):
        board = chess.Board()
        network_color = chess.WHITE if starting_player == "network" else chess.BLACK
        self.other_engine.board = board

        while not board.is_game_over():
            if board.turn == network_color:
                # Network's turn: get best move from raw network output
                board_tensor = self.converter.board_to_input_tensor(board)
                policy, value = self.network.predict(board_tensor)
                move_probs = self.converter.mask_illegal_moves(board, policy)
                best_index = tf.argmax(move_probs).numpy()
                move_uci = self.converter._get_move_from_index(best_index)

                # Mirror back if network played as black
                if board.turn == chess.BLACK:
                    move_uci = mirror_move_uci(move_uci)

                move = chess.Move.from_uci(move_uci)
            else:
                move = self.other_engine.choose_random_move()

            board.push(move)
            print(move)


        result = board.result()
        if result == "1/2-1/2":
            return 0.5
        elif (result == "1-0" and network_color == chess.WHITE) or \
            (result == "0-1" and network_color == chess.BLACK):
            return 1
        else:
            return 0