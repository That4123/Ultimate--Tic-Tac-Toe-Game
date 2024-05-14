import pygame, sys
from pygame.locals import K_TAB, QUIT, K_RIGHT
from state import State, State_2, UltimateTTT_Move
import time
from importlib import import_module
import numpy as np
import tensorflow as tf
import keras
from keras import layers
from keras.models import Model
from keras import models
from keras.optimizers import Adam
import math
from collections import deque
import os
from datetime import datetime
import copy


color = {"black": pygame.Color(0, 0, 0),
         "white": pygame.Color(255, 255, 255),
         'blue': pygame.Color(50, 255, 255),
         'orange': pygame.Color(255, 120, 0)
        }
small_image = {1: pygame.image.load('images/small_x.png'), 
               -1: pygame.image.load('images/small_o.png')}
large_image = {1: pygame.image.load('images/large_x.png'), 
               -1: pygame.image.load('images/large_o.png')}

pygame.init()

screen = pygame.display.set_mode((600, 600))
pygame.display.set_caption('Ultimate Tic-Tac-Toe')

def draw(state: State_2):
    screen.fill('white')
        
    for x in range(3):
        for y in range(3):
            pygame.draw.rect(screen, color["white"], (x*200, y*200, 200, 200))

    if state.previous_move != None:
        next_block = state.previous_move.x * 3 + state.previous_move.y
        pygame.draw.rect(screen, color['blue'], 
              ((next_block%3)*200, (next_block//3)*200, 200, 200))

        i = state.previous_move.index_local_board
        pygame.draw.rect(screen, color['orange'],(
                         (i%3)*200 + state.previous_move.y*50 + 25,
                         (i//3)*200 + state.previous_move.x*50 + 25,
                         50,50))
    
    for k in range(9):
        value = state.global_cells[k]
        if value != 0:
            picture = large_image[value]
            picture = pygame.transform.scale(picture, (100, 100))
            screen.blit(picture,((k%3)*200 + 50,(k//3)*200 + 50))            
    
    for x in range(3):
        for y in range(3):
            for i in [1,2]:
                pygame.draw.line(screen, color["black"], 
                                 (x*200 + i*50 + 25, y*200 + 25), 
                                 (x*200 + i*50 + 25, y*200 + 175), 2)
                pygame.draw.line(screen, color["black"], 
                                 (x*200 + 25, y*200 + i*50 + 25), 
                                 (x*200 + 175, y*200 + i*50 + 25), 2)
    
    for i in range(9):
        local_board = state.blocks[i]
        for x in range(3):
            for y in range(3):
                value = local_board[x, y]
                if value != 0:
                    screen.blit(small_image[value],
                                ((i%3)*200 + y*50 + 35,
                                (i//3)*200 + x*50 + 35))
    
    for i in [1, 2]:
        pygame.draw.line(screen, color["black"], (i*200, 0), (i*200, 600), 3)
        pygame.draw.line(screen, color["black"], (0, i*200), (600, i*200), 3)

    pygame.display.update()
    

def play_step_by_step(player_X, player_O, rule = 1):
    player_1 = import_module(player_X)
    player_2 = import_module(player_O)
    if rule == 1:
        state = State()
    else:
        state = State_2()
    turn = 0
    remain_time_X = 120
    remain_time_O = 120
    is_game_done = False
 
    while True:
        draw(state)
        
        for event in pygame.event.get():          
            if event.type == QUIT:
                pygame.quit()
                sys.exit()     
            
            if state.game_over or is_game_done:
                continue
            
            if event.type == pygame.KEYDOWN:
                if event.key == K_TAB or event.key == K_RIGHT:
                    
                    start_t = time.time()
                    if state.player_to_move == 1:
                        new_move = player_1.select_move(state, remain_time_X)
                        elapsed_time = time.time() - start_t
                        remain_time_X -= elapsed_time
                    else:
                        new_move = player_2.select_move(state, remain_time_O)
                        elapsed_time = time.time() - start_t
                        remain_time_O -= elapsed_time
                    
                    if elapsed_time > 10 or not new_move:
                        is_game_done = True
                    
                    if (remain_time_O < -0.1) or (remain_time_X < -0.1):
                        is_game_done = True
                        
                    state.act_move(new_move)
                    turn += 1
                    if turn == 81:
                        is_game_done = True
                    

def play_auto(player_X, player_O, rule = 1):
    player_1 = import_module(player_X)
    player_2 = import_module(player_O)
    if rule == 1:
        state = State()
    else:
        state = State_2()
    turn = 0
    remain_time_X = 120
    remain_time_O = 120
    is_game_done = False
    
    while True:
        draw(state)
        # delay drawing
        # time.sleep(1)
        
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()         
        
        if state.game_over or is_game_done:
            break

        start_t = time.time()
        if state.player_to_move == 1:
            new_move = player_1.select_move(state, remain_time_X)
            elapsed_time = time.time() - start_t
            remain_time_X -= elapsed_time
        else:
            new_move = player_2.select_move(state, remain_time_O)
            elapsed_time = time.time() - start_t
            remain_time_O -= elapsed_time
        
        if elapsed_time > 10 or not new_move or \
                (remain_time_O < -0.1) or (remain_time_X < -0.1):
            is_game_done = True
            continue
        
        state.act_move(new_move)
        turn += 1
    
    
train_episodes = 20
mcts_search = 20
n_pit_network = 20
playgames_before_training = 2
cpuct = 4
training_epochs = 2
learning_rate = 0.001
save_model_path = 'training_dir'

# initializing search tree
Q = {}  # state-action values
Nsa = {}  # number of times certain state-action pair has been visited
Ns = {}   # number of times state has been visited
W = {}  # number of total points collected after taking state action pair
P = {}  # initial predicted probabilities of taking certain actions in state

def neural_network():
    
    input_layer = layers.Input(shape=(9,9), name="BoardInput")
    reshape = layers.Reshape((9,9,1))(input_layer)
    conv_1 = layers.Conv2D(128, (3,3), padding='valid', activation='relu', name='conv1')(reshape)
    conv_2 = layers.Conv2D(128, (3,3), padding='valid', activation='relu', name='conv2')(conv_1)
    conv_3 = layers.Conv2D(128, (3,3), padding='valid', activation='relu', name='conv3')(conv_2)

    conv_3_flat = layers.Flatten()(conv_3)

    dense_1 = layers.Dense(512, activation='relu', name='dense1')(conv_3_flat)
    dense_2 = layers.Dense(256, activation='relu', name='dense2')(dense_1)

    pi = layers.Dense(81, activation="softmax", name='pi')(dense_2)
    v = layers.Dense(1, activation="tanh", name='value')(dense_2)

    model = Model(inputs=input_layer, outputs=[pi, v])
    model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=Adam(learning_rate))

    model.summary()
    return model
nn = neural_network()
def mcts(state, turn=0):
    if turn >= 7: return 0
    if state.previous_move == None:
        possibleA = range(81)
    else:
        possibleA = [] 
        valid_moves = state.get_valid_moves
        for move in valid_moves:
            possibleA.append(translate(move))    
    # sArray = board_to_array(s, mini_board, current_player)
    sArray = get_state(state, state.player_to_move)
    sTuple = tuple(map(tuple, sArray))

    if len(possibleA) > 0:
        if sTuple not in P.keys():

            policy, v = nn.predict(sArray.reshape(1,9,9))
            v = v[0][0]
            valids = np.zeros(81)
            np.put(valids,possibleA,1)
            policy = policy.reshape(81) * valids
            policy = policy / np.sum(policy)
            P[sTuple] = policy

            Ns[sTuple] = 1

            for a in possibleA:
                Q[(sTuple,a)] = 0
                Nsa[(sTuple,a)] = 0
                W[(sTuple,a)] = 0
            return -v

        best_uct = -100
        for a in possibleA:

            uct_a = Q[(sTuple,a)] + cpuct * P[sTuple][a] * (math.sqrt(Ns[sTuple]) / (1 + Nsa[(sTuple,a)]))

            if uct_a > best_uct:
                best_uct = uct_a
                best_a = a
        
        # next_state, mini_board, wonBoard = move(s, best_a, current_player)
        new_state = copy.deepcopy(state)
        new_state.act_move(translate_move(best_a,state.player_to_move))
        wonBoard = new_state.game_over

        if wonBoard:
            v = 1
        else:
            v = mcts(new_state, turn+1)
    else:
        return 0

    W[(sTuple,best_a)] += v
    Ns[sTuple] += 1
    Nsa[(sTuple,best_a)] += 1
    Q[(sTuple,best_a)] = W[(sTuple,best_a)] / Nsa[(sTuple,best_a)]
    return -v

def get_action_probs(state):

    for _ in range(mcts_search):
        new_state = copy.deepcopy(state)
        value = mcts(new_state)
    
    print ("done one iteration of MCTS")

    actions_dict = {}

    sArray = get_state(state, state.player_to_move)
    sTuple = tuple(map(tuple, sArray))
    possibleA = [] 
    valid_moves = state.get_valid_moves
    for move in valid_moves:
        possibleA.append(translate(move)) 

    for a in possibleA:
        actions_dict[a] = Nsa[(sTuple,a)] / Ns[sTuple]
    
    action_probs = np.zeros(81)
    
    for a in actions_dict:
        np.put(action_probs, a, actions_dict[a], mode='raise')
    
    return action_probs
def possiblePos(state):
    possibleA = [] 
    valid_moves = state.get_valid_moves
    for move in valid_moves:
        possibleA.append(translate(move)) 
    return possibleA
def playgame():
    done = False
    state = State_2()
    turn = 0
    game_mem = []
    is_game_done = False
    while not done:
        draw(state)
        policy = get_action_probs(state)
        policy = policy / np.sum(policy)
        # game_mem.append([board_to_array(real_board, mini_board, current_player), current_player, policy, None])
        game_mem.append([get_state(state, state.player_to_move),state.player_to_move, policy,None])
        action = np.random.choice(len(policy), p=policy)
        
        # print ("policy ", policy)
        # print ("chosen action", action)
        # print ("mini-board", mini_board)
        # print_board(real_board)

        # next_state, mini_board, wonBoard = move(real_board, action, current_player)
        state.act_move(translate_move(action,state.player_to_move))
        wonBoard = state.game_over
        if len(possiblePos(state)) == 0:
            for tup in game_mem:
                tup[3] = 0
            return game_mem

        if wonBoard:
            for tup in game_mem:
                if tup[1] == state.player_to_move*-1:
                    tup[3] = 1
                else:
                    tup[3] = -1
            return game_mem
def train_nn(nn, game_mem):

    print ("Training Network")
    print ("length of game_mem", len(game_mem))
    
    state = []
    policy = []
    value = []

    for mem in game_mem:
        state.append(mem[0])
        policy.append(mem[2])
        value.append(mem[3])
    
    state = np.array(state)
    policy = np.array(policy)
    value = np.array(value)
    
    
    history = nn.fit(state, [policy, value], batch_size=32, epochs=training_epochs, verbose=1)

def pit(nn, new_nn):

    # function pits the old and new networks. If new netowork wins 55/100 games or more, then return True
    print ("Pitting networks")
    nn_wins = 0
    new_nn_wins = 0

    for _ in range(n_pit_network):

        done = False
        state = State_2()
        turn = 0
        game_mem = []
        is_game_done = False

        while True:
            draw(state)
            policy, v = nn.predict(get_state(state, state.player_to_move).reshape(1,9,9))
            valids = np.zeros(81)

            possibleA = possiblePos(state)
            
            if len(possibleA) == 0:
                break

            np.put(valids,possibleA,1)
            policy = policy.reshape(81) * valids
            policy = policy / np.sum(policy)
            action = np.argmax(policy)

            # next_state, mini_board, win = move(s, action, 1)
            state.act_move(translate_move(action,state.player_to_move))
            win = state.game_over

            if win:
                nn_wins +=1
                break


            # new nn makes move

            policy, v = new_nn.predict(get_state(state, state.player_to_move).reshape(1,9,9))
            valids = np.zeros(81)

            possibleA = possiblePos(state)
            
            if len(possibleA) == 0:
                break

            np.put(valids,possibleA,1)
            policy = policy.reshape(81) * valids
            policy = policy / np.sum(policy)
            action = np.argmax(policy)

            state.act_move(translate_move(action,state.player_to_move))
            win = state.game_over
            if win:
                new_nn_wins += 1
                break
    
    if (new_nn_wins + nn_wins) == 0:
        print ("The game was a complete tie")
        now = datetime.utcnow()
        filename = 'tictactoeTie{}.h5'.format(now)
        model_path = os.path.join(save_model_path,filename)
        nn.save(model_path)
        return False

    win_percent = float(new_nn_wins) / float(new_nn_wins + nn_wins)
    if win_percent > 0.52:
        print ("The new network won")
        print (win_percent)
        return True
    else:
        print ("The new network lost")
        print (new_nn_wins)
        return False
def train():

    global nn
    global Q
    global Nsa
    global Ns
    global W
    global P

    game_mem = []

    for episode in range(train_episodes):
        
        nn.save('temp.h5')
        old_nn = models.load_model('temp.h5')

        for _ in range(playgames_before_training):
            game_mem += playgame()
        
        train_nn(nn, game_mem)
        game_mem = []
        if pit(old_nn, nn):
            del old_nn
            Q = {}
            Nsa = {}
            Ns = {}
            W = {}
            P = {}
        else:
            nn = old_nn
            del old_nn


    now = datetime.utcnow()
    filename = 'tictactoe_MCTS200{}.h5'.format(now)
    model_path = os.path.join(save_model_path,filename)
    nn.save(model_path)
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
def translate_move(move, player_to_move):
    b = move//9
    c = move%9 
    x = b %3
    y =c% 3
    m = b//3
    n = c//3
    a = m*3+n

    new_move = UltimateTTT_Move(a, x, y, player_to_move)
    return new_move
def get_state( game, player_to_move):

        state = copy.deepcopy(game.blocks)
        for i in range (9):
            if game.global_cells[i] != 0:
                for j in range (3):
                    for k in range (3):
                        state[i][j,k] = game.global_cells[i] * player_to_move
            else:
                for j in range (3):
                    for k in range (3):
                        state[i][j,k] = game.blocks[i][j,k] * player_to_move
        if game.previous_move == None:
                for i in range (9):
                    for j in range (3):
                        for k in range                                                  (3):
                            state[i][j,k] = 0.1
        else:
            count_valid = 0
            local_board = (game.previous_move.x)*3 + game.previous_move.y
            for j in range (3):
                    for k in range (3):
                        if (state[local_board][j,k]==0):
                            state[local_board][j,k] = 0.1                            
                            count_valid += 1
            if count_valid == 0:
                for i in range (9):
                    for j in range (3):
                        for k in range (3):
                            if (state[i][j,k]==0):
                                state[i][j,k] = 0.1
        array = []
        for i in range (9):
            if i%3 == 0:
                for j in range (3):
                    m=[]
                    for l in range (3):
                        for k in range (3):
                            m.append(state[i+l][j,k])
                    array.append(m)
        return np.array(array)
def translate(move):
    a = move.index_local_board
    x= move.x
    y= move.y
    m = a//3
    n = a%3
    b = m*3+x
    c = n*3 + y
    return b*9+c
def translate_2(a,x,y):
    m = a//3
    n = a%3
    b = m*3+x
    c = n*3 + y
    return b*9+c

def play_fight(player_X):
    
    no_win = 0
    no_lose = 0
    no_draw = 0
    nn = models.load_model('temp.h5')
    for i in range (10):
        player_1 = import_module(player_X)
        state = State_2()
        turn = 0
        remain_time_X = 120
        remain_time_O = 120
        is_game_done = False
        while True:
            draw(state)
            # delay drawing
            # time.sleep(1)
            
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()         
            
            if state.game_over or is_game_done:
                result = state.game_result(state.global_cells.reshape(3,3))
                if(result==-1): no_win +=1
                elif (result==-1): no_lose +=1
                else: 
                    if state.count_O > state.count_X : no_win+=1
                    elif state.count_O < state.count_X: no_lose+=1
                    else: 
                        no_draw +=1
                break

            start_t = time.time()
            if state.player_to_move == 1:
                new_move = player_1.select_move(state, remain_time_X)
                elapsed_time = time.time() - start_t
                remain_time_X -= elapsed_time
                if (new_move == None): 
                    is_game_done = True
                    continue
                state.act_move(new_move)
            else:
                policy, v = nn.predict(get_state(state, state.player_to_move).reshape(1,9,9))
                valids = np.zeros(81)

                possibleA = possiblePos(state)
                if len(possibleA) == 0:
                    is_game_done = True
                    continue

                np.put(valids,possibleA,1)
                policy = policy.reshape(81) * valids
                policy = policy / np.sum(policy)
                action = np.argmax(policy)

                # next_state, mini_board, win = move(s, action, 1)
                state.act_move(translate_move(action,state.player_to_move))
                elapsed_time = time.time() - start_t
                remain_time_O -= elapsed_time
            
            if elapsed_time > 10 or not new_move or \
                    (remain_time_O < -0.1) or (remain_time_X < -0.1):
                is_game_done = True
                continue
            turn += 1
    print("NUMBER OF WIN IS: ", no_win)
    print("NUMBER OF LOSE IS: ", no_lose)
    print("NUMBER OF DRAW IS: ",no_draw)
play_fight("random_agent")