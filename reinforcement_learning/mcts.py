import math
from random import random

from reinforcement_learning.board import Board


class MCTS:

    def __init__(self, network):
        self.network = network
        self.root_node = None
        self.tau = 1.0  # Temperature parameter for exploration
        self.c_puct = 1.0  # Exploration constant for UCT formula
        self.mcts_iterations = 100  # Number of MCTS iterations per move

    def uct_value(self, edge, parent_N):
        # Important for Selection - select edge that maximeses edge.Q + uct_value(edge)
        return self.c_puct * edge.P * (math.sqrt(parent_N) / (1 + edge.N))

    def select(self, node):
        if node.is_leaf():
            return node
        else:
            max_uct_child = None
            max_uct_value = float("-inf")
            for edge, child_node in node.child_edge_node:
                uct_val = self.uct_value(edge, edge.parent_node.parent_edge.N)
                val = edge.Q
                if edge.parent_node.board.turn == Board.BLACK:
                    val = -edge.Q  # Negate value for opponent's turn
                uct_val_child = val + uct_val
                if uct_val_child > max_uct_value:
                    max_uct_child = child_node
                    max_uct_value = uct_val_child
            all_best_childs = []
            for edge, child_node in node.child_edge_node:
                uct_val = self.uct_value(edge, edge.parent_node.parent_edge.N)
                val = edge.Q
                if edge.parent_node.board.turn == Board.BLACK:
                    val = -edge.Q  # Negate value for opponent's turn
                uct_val_child = val + uct_val
                if uct_val_child == max_uct_value:
                    all_best_childs.append(child_node)
                if max_uct_child is None:
                    raise ValueError("Could not identify child with best uct value.")
                else:
                    if len(all_best_childs) > 1:
                        idx = random.randint(0, len(all_best_childs) - 1)
                        return self.select(all_best_childs[idx])
                    else:
                        return self.select(max_uct_child)

    def expand_and_evaluate(self, node):
        terminal, winner = node.board.is_terminal()
        if terminal:
            value = 0.0
            if winner == Board.WHITE:
                value = 1.0
            elif winner == Board.BLACK:
                value = -1.0
            self.backpropagate(value, node.parent_edge)
            return
        value = node.expand(self.network)
        self.backpropagate(value, node.parent_edge)

    def backpropagate(self, value, edge):
        edge.N += 1
        edge.W += value
        edge.Q = edge.W / edge.N
        if edge.parent_node != None:
            self.backpropagate(value, edge.parent_node.parent_edge)

    def search(self, root_node):
        self.root_node = root_node
        _ = self.root_node.expand(
            self.network
        )  # Initialize root node with network evaluation
        for i in range(self.mcts_iterations):  # Number of MCTS iterations
            selected_node = self.select(root_node)
            self.expand_and_evaluate(selected_node)
        N_sum = 0
        move_probabilities = []
        for edge, _ in root_node.child_edge_node:
            N_sum += edge.N
        for edge, node in root_node.child_edge_node:
            probability = (edge.N ** (1 / self.tau)) / ((N_sum) ** (1 / self.tau))
            move_probabilities.append((edge.move, probability, edge.N, edge.Q))
        return move_probabilities
