import random


class Node:
    def __init__(self, move=None, parent=None):
        self.move = move
        self.label = f"{LAYER}-{move}" if move else "root"
        self.parent = parent
        self.children = []
        self.Q = 0  # Total move value of this node
        self.W = 0  # Total sum of move values
        self.N = 0  # Visit count
        self.P = 0  # Prior probability from the network
        self.u = 0  # Exploration term

    def is_unvisited(self):
        return self.N == 0

    def update_Q_W_N(self, value):
        self.W += value
        self.N += 1
        self.Q = self.W / self.N


def generate_mock_network_outputs():
    outputs_size = 50
    # generate 50 numbers between 0 and 1, summing to 1
    # make it a dict to correspond to possible moves, naming moves as "move1", "move2", ..., "move50"
    policy = {f"move{i+1}": random.random() for i in range(outputs_size)}
    total = sum(policy.values())
    policy = {move: prob / total for move, prob in policy.items()}
    # generate a random value between -1 and 1
    value = random.uniform(-1, 1)
    return policy, value


def generate_mock_possible_moves():
    print("Generating legal moves for the position...")
    num_moves = random.randint(2, 5)  # Random number of possible moves
    # create list naming moves as "move1", "move2", ..., "moveN"
    possible_moves = [f"move{i+1}" for i in range(num_moves)]
    print(f"Possible moves are: {possible_moves}")
    return possible_moves


def query_network(position: str, possible_moves: list[str]):
    print(f"Querying network for position: {position}")
    policy, value = generate_mock_network_outputs()
    # print(
    #    "Policy:", {move: round(prob, 4) for move, prob in list(policy.items())[:5]}
    # )  # Print first 5 policy values for brevity
    # print(f"Value for {position}:", value)

    # mask policy to recalculate probabilites only for possible moves, adding to 1 again
    masked_policy = {
        move: prob for move, prob in policy.items() if move in possible_moves
    }
    total_masked = sum(masked_policy.values())
    masked_policy = {move: prob / total_masked for move, prob in masked_policy.items()}
    print(
        "Masked Policy (only legal moves):",
        {move: round(prob, 4) for move, prob in masked_policy.items()},
    )
    return masked_policy, value


def make_move(node):
    # Placeholder for move application logic
    print(f"Applying move: {node.label}")
    return f"new_position_after_{node.label}"  # Mock new position representation


def calculate_u(node):
    c = 1.0  # Exploration constant
    parent_possible_moves = len(node.parent.children) if node.parent else 1
    u = c * node.P * (parent_possible_moves**0.5) / (1 + node.N)
    print(f"{node.label} - u: {u:.4f}")
    return u


def select_child_that_maximizes_Q_plus_u(node):
    print("SELECTION ---------------------")
    for child in node.children:
        child.u = calculate_u(child)
    chosen_child = max(node.children, key=lambda n: n.Q + n.u)
    print(
        f"Selected child {chosen_child.label} with Q={chosen_child.Q:.4f} and u={chosen_child.u:.4f}"
    )
    return chosen_child


def backpropagate(node, value):
    print("BACKPROPAGATION ---------------------")
    print(f"Backpropagating value {value:.4f} from node {node.label} up the tree...")
    current_node = node
    while current_node is not None:
        current_node.update_Q_W_N(value)
        print(
            f"Updated node {current_node.label}: W={current_node.W:.4f}, N={current_node.N}, Q={current_node.Q:.4f}"
        )
        current_node = current_node.parent


def expand(node):
    print("EXPANDING ---------------------")
    print(f"Expanding node {node.label}...")
    new_position = make_move(node)  # Placeholder for move application
    new_possible_moves = generate_mock_possible_moves()  # Generate new possible moves
    masked_policy, value = query_network(new_position, new_possible_moves)
    print(f"{node.label} resulted in a position, yielding value: {value:.4f}")
    print(f"Generating children for {node.label} based on new possible moves...")
    for move in new_possible_moves:
        grandchild_node = Node(move=move, parent=node)
        grandchild_node.P = masked_policy[move]  # Set prior probability
        node.children.append(grandchild_node)
    print(
        f"""Node {node.label} has been expanded with {len(new_possible_moves)} children based on the new position's possible moves."""
    )
    return value  # Return the value for backpropagation


def select_move_given_policy(node):
    # $\pi_m = \dfrac{N^{1/ \tau}_m}{\displaystyle \sum^n N^{1/ \tau}_m}$
    print("MOVE SELECTION ---------------------")
    print(f"Selecting move from root node based on visit counts...")
    TAU = 0.9  # Temperature parameter for exploration
    visit_counts = [child.N for child in node.children]
    total_visits = sum(visit_counts)
    if total_visits == 0:
        # If no visits, select a random child
        chosen_child = random.choice(node.children)
    else:
        # Calculate the probability distribution
        probabilities = [visit ** (1 / TAU) / total_visits for visit in visit_counts]
        # Choose a child based on the probability distribution
        chosen_child = random.choices(node.children, weights=probabilities)[0]
    print(
        f"Selected move: {chosen_child.label} with visit count {chosen_child.N} and probability {probabilities[node.children.index(chosen_child)]:.4f}"
    )
    return chosen_child  # Return the move associated with the selected child


# - Q is move value
# - W is a helper variable that stores the sum of move values
# - N is the visit count
# - P is the prior probability from the network's policy head
Q = W = N = 0
LAYER = 1  # To track the depth of the tree for labeling purposes

print("Initializing MCTS with root node...")
root_node = Node()
possible_moves = generate_mock_possible_moves()
masked_policy, value = query_network("initial_position", possible_moves)
for move in possible_moves:
    child_node = Node(move=move, parent=root_node)
    child_node.P = masked_policy[move]  # Set prior probability from masked policy
    root_node.children.append(child_node)
LAYER += 1


# chosen child is the one with the highest Q + u
chosen_child = select_child_that_maximizes_Q_plus_u(root_node)

if chosen_child.is_unvisited():
    print(
        f"\nExploring unvisited node: {chosen_child.label} with P={chosen_child.P:.4f}"
    )
    value = expand(chosen_child)  # Placeholder for expansion logic
    print(f"Backpropagating")
    backpropagate(chosen_child, value)  # Placeholder for backpropagation logic

else:
    print(f"\nSelected node {chosen_child.label} has been visited before")

# After running the MCTS iterations, select the move to play based on visit counts
chosen_child_given_policy = select_move_given_policy(root_node)
