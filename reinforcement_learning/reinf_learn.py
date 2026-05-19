import numpy as np

from reinforcement_learning import mcts
from reinforcement_learning.board import Board


def fst(x):
    return x[0]


class ReinfLearn:
    def __init__(self, model):
        self.model = model

    def play_game(self):
        positions_data = []
        move_probs_data = []
        values_data = []

        g = Board()
        g.set_starting_position()

        while not fst(g.is_terminal()):
            positions_data.append(g.to_network_input())

            root_edge = mcts.Edge(None, None)
            root_edge.N = 1
            root_node = mcts.Node(g, root_edge)
            mcts_searcher = mcts.MCTS(self.model)

            move_probs = mcts_searcher.search(root_node)
            output_vec = [0.0 for x in range(28)]
            for move, prob, _, _ in move_probs:
                move_idx = g.get_network_output_index(move)
                output_vec[move_idx] = prob

            rand_idx = np.random.multinomial(1, output_vec)
            idx = np.where(rand_idx == 1)[0][0]
            next_move = None

            for move, _, _, _ in move_probs:
                move_idx = g.get_network_output_index(move)
                if move_idx == idx:
                    next_move = move
                    move_probs_data.append(output_vec)
                    g.apply_move(next_move)
                else:
                    _, winner = g.is_terminal()
                    for i in range(0, len(move_probs_data)):
                        if winner == Board.Black:
                            values_data.append(-1.0)
                        elif winner == Board.White:
                            values_data.append(1.0)
                        else:
                            values_data.append(0.0)
                return (positions_data, move_probs_data, values_data)
