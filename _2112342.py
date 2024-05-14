import numpy as np
import math
import copy
from state import UltimateTTT_Move 
max_depth = 4
def select_move(cur_state, remain_time):
    if cur_state.previous_move == None: return UltimateTTT_Move(4,1,1,1)
    player = cur_state.player_to_move
    if remain_time > 10: 
        move = get_best_move(cur_state, player)
        if move != None: 
            return move
    else: 
        valid_moves = cur_state.get_valid_moves
        if len(valid_moves) != 0:
            print ('Play random')
            return np.random.choice(valid_moves)
def get_best_move(state, player):
    _, best_move = maximize(state, -math.inf, math.inf, 0, player)
    return best_move

def maximize(state, alpha, beta, depth, player):
    if state.game_over:
        return evaluate(state, depth, player), None
    max_eval = -math.inf
    best_move = None
    if depth == max_depth:
        max_eval = evaluate (state, depth, player)  
        return max_eval, best_move
    valid_moves = state.get_valid_moves
    if len(valid_moves) != 0:
        for move in valid_moves:
            new_state = copy.deepcopy(state)
            new_state.act_move(move)
            eval, _ = minimize(new_state, alpha, beta, depth + 1, player)

            if eval > max_eval:
                max_eval = eval
                best_move = move

            alpha = max(alpha, eval)
            if beta <= alpha:
                break

    return max_eval, best_move

def minimize(state, alpha, beta, depth, player):
    if state.game_over:
        return  evaluate(state, depth,player), None

    min_eval = math.inf
    best_move = None
    if depth == max_depth:
        min_eval = evaluate (state, depth, player)  
        return min_eval, best_move
    valid_moves = state.get_valid_moves
    if len(valid_moves) != 0:
        for move in valid_moves:
            new_state = copy.deepcopy(state)
            new_state.act_move(move)
            eval, _ = maximize(new_state, alpha, beta, depth + 1, player)

            if eval < min_eval:
                min_eval = eval
                best_move = move

            beta = min(beta, eval)
            if beta <= alpha:
                break

    return min_eval, best_move
def evaluate(state, depth, player):
    score = 0
    if state.game_over:
        result = state.game_result(state.global_cells.reshape(3,3))
        if result == 0: 
            return 0
        else:
            score = result*player*100 - depth*3 
        return score 
    score += (state.count_X - state.count_O)*player*20 - depth*3
    index_local_board = state.previous_move.x * 3 + state.previous_move.y
    score += check_local_board(state.blocks[index_local_board],player)
    score += check_local_board(state.blocks[state.previous_move.index_local_board],player)
    
    return score 
def check_local_board(board, player):
        row_sum = np.sum(board, 1)
        col_sum = np.sum(board, 0)
        diag_sum_topleft = board.trace()
        diag_sum_topright = board[::-1].trace()
        value = 0
        value += (any(row_sum* player >= 2) + any(col_sum *player >= 2))*5
        value += ((diag_sum_topleft *player >= 2) + (diag_sum_topright*player >= 2)) *5
        value -= (any(row_sum* player <= -2) + any(col_sum*player <= -2))*4
        value -= ((diag_sum_topleft*player <= -2) + (diag_sum_topright*player <= -2))*4   
        if np.all(board != 0):
            return 0
        return value
