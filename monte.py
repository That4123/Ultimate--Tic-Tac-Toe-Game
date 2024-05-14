import random
import math
import copy 
class UCTNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

def UCT(node):
    if node.visits == 0:
        return math.inf
    exploration_weight = 1.0
    exploitation = node.value / node.visits
    exploration = exploration_weight * math.sqrt(math.log(node.parent.visits) / node.visits)
    return exploitation + exploration

def select_best_child(node):
    return max(node.children, key=UCT)

def expand(node):
    valid_moves = node.state.get_valid_moves
    if len(valid_moves) != 0:
        move = random.choice(valid_moves)
        new_state = copy.deepcopy(node.state)  # Assume you have a clone method
        new_state.act_move(move)
        new_node = UCTNode(new_state, parent=node)
        node.children.append(new_node)
        return new_node
    else: 
        return None

def simulate_random_play(node):
    state = copy.deepcopy(node.state)
    while not state.game_over:
        valid_moves = state.get_valid_moves
        if len(valid_moves) != 0:
            move = random.choice(valid_moves)
            state.act_move(move)
        else: 
            return 0
    return state.game_result(state.global_cells.reshape(3, 3))

def backpropagate(node, result):
    while node is not None:
        node.visits += 1
        node.value += result
        node = node.parent

def monte_carlo_search(root, num_iterations):
    for _ in range(num_iterations):
        node = root
        while node.children:
            node = select_best_child(node)
        if node.visits > 0:
            node = expand(node)
        if node == None:  break 
        result = simulate_random_play(node)
        backpropagate(node, result)
    best_child = max(root.children, key=lambda x: x.visits)
    return best_child.state.previous_move
