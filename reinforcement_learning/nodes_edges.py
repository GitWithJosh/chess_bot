"Alpha-Zero like Reinforcement Learning with Monte Carlo Tree Search for Chess"

import copy
import chess

from converter import Converter


class Edge:
    def __init__(self, move, parent_node):
        self.parent_node = parent_node
        self.move = move
        self.N = 0  # Visit count
        self.W = 0  # Total value of this move
        self.Q = 0  # Value of this move
        self.P = 0  # Prior probability from the network


class Node:
    def __init__(self, board:chess.Board, parent_edge):
        self.board = board
        self.parent_edge = parent_edge
        self.child_edge_node = []

    def expand(self, network):

        for move in self.board.legal_moves:
            child_board = copy.deepcopy(self.board)
            child_board.push(chess.Move.from_uci(move))
            child_edge = Edge(move, self)
            child_node = Node(child_board, child_edge)
            self.child_edge_node.append((child_edge, child_node))
        # Query the network for the child node's position and set the prior probability
        # Return is of shape policy(1,28), value(1,1)
        # Therefore q[0] = policy array, q[1] = value array
        q = network.predict(self.board.to_network_input())
        prob_sum = 0.0
        for edge, _ in self.child_edge_node:
            move_index = self.board.get_network_output_index(edge.move)
            edge.P = q[0][0][move_index]
            prob_sum += edge.P
        # Normalize the prior probabilities
        for edge, _ in self.child_edge_node:
            edge.P /= prob_sum if prob_sum > 0 else 1
        value = q[1][0][0]
        return value

    def is_leaf(self):
        return self.child_edge_node == []