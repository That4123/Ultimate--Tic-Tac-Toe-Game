import torch
import random
import numpy as np
from collections import deque
from state import State, State_2
from model import Linear_QNet, QTrainer
from helper import plot
from importlib import import_module

MAX_MEMORY = 100_000
BATCH_SIZE = 200
LR = 0.001

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(81, 5184, 81)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game, player_to_move):

        state = np.zeros(81)
        for i in range (9):
            if game.global_cells[i] != 0:
                for j in range (3):
                    for k in range (3):
                        state[translate_2(i,j,k)] = game.global_cells[i] * player_to_move
            else:
                for j in range (3):
                    for k in range (3):
                        state[translate_2(i,j,k)] = game.blocks[i][j,k] * player_to_move
        if game.previous_move == None:
                for j in range (3):
                    for k in range (3):
                        state[translate_2(i,j,k)] = 2
        else:
            count_valid = 0
            local_board = (game.previous_move.x)*3 + game.previous_move.y
            for j in range (3):
                    for k in range (3):
                        if (state[translate_2(local_board,j,k)]==0):
                            state[translate_2(local_board,j,k)] = 2
                            count_valid += 1
            if count_valid == 0:
                for i in range (9):
                    for j in range (3):
                        for k in range (3):
                            if (state[translate_2(i,j,k)]==0):
                                state[translate_2(i,j,k)] = 2
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, game):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 100 - self.n_games//10
        move = None
        if random.randint(0, 200) < self.epsilon:
            valid_moves = game.get_valid_moves
            if len(valid_moves) != 0:
                new_move = np.random.choice(valid_moves)
                move = translate(new_move)
            else: 
                return -1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()

        return move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = State_2()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
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

if __name__ == '__main__':
    train()