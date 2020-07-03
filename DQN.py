# Imports
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
import random
import numpy as np
import pandas as pd
from operator import add
import collections


# Deep Q-learning network ----------------------------------------------------------------------------------------------
class DQNAgent(object):
    # Network initialisation
    def __init__(self, par):
        # define_parameters inputs
        self.first_layer = par['first_layer_size']
        self.second_layer = par['second_layer_size']
        self.third_layer = par['third_layer_size']

        self.memory = collections.deque(maxlen=par['memory_size'])
        self.weights = par['weights_path']
        self.load_weights = par['load_weights']

        # currently 0.0005 to decay
        self.learning_rate = par['learning_rate']
        self.epsilon = 1
        self.actual = []

        self.r = 0
        self.gamma = 0.9

        self.dataframe = pd.DataFrame()
        self.short_memory = np.array([])

        self.agent_target = 1
        self.agent_predict = 0

        # make the neural network model
        self.model = self.network()

    # network ----------------------------------------------------------------------------------------------------------
    def network(self):
        # linear stack of layers  = sequential model
        model = Sequential()

        # hidden layers with 11 input parameters
        # output_dim=self.first_layer etc.
        model.add(Dense(self.first_layer, activation='relu', input_dim=11))
        model.add(Dense(self.second_layer, activation='relu'))
        model.add(Dense(self.third_layer, activation='relu'))

        # output layer
        # output_dim=3
        model.add(Dense(3, activation='softmax'))

        # compiles the optimizer
        opt = Adam(self.learning_rate)
        model.compile(loss='mse', optimizer=opt)

        # inputs weights
        if self.load_weights:
            model.load_weights(self.weights)
        return model

    # States possible --------------------------------------------------------------------------------------------------
    def get_state(self, game, snake, food):
        # possible states entered
        st = [
            # danger straight
            (snake.x_ch == 20 and
             snake.y_ch == 0 and
             ((list(map(add, snake.pos[-1], [20, 0]))
               in snake.pos) or
              snake.pos[-1][0] + 20 >= (game.game_w - 20)))
            or (snake.x_ch == -20 and
                snake.y_ch == 0 and
                ((list(map(add, snake.pos[-1], [-20, 0]))
                  in snake.pos) or
                 snake.pos[-1][0] - 20 < 20))
            or (snake.x_ch == 0 and
                snake.y_ch == -20 and
                ((list(map(add, snake.pos[-1], [0, -20]))
                  in snake.pos) or
                 snake.pos[-1][-1] - 20 < 20))
            or (snake.x_ch == 0 and
                snake.y_ch == 20 and
                ((list(map(add, snake.pos[-1], [0, 20]))
                  in snake.pos) or snake.pos[-1][-1] + 20 >=
                 (game.game_h - 20))),

            # danger right
            (snake.x_ch == 0 and
             snake.y_ch == -20 and
             ((list(map(add, snake.pos[-1], [20, 0]))
               in snake.pos) or
              snake.pos[-1][0] + 20 > (game.game_w - 20))) or
            (snake.x_ch == 0 and
             snake.y_ch == 20 and
             ((list(map(add, snake.pos[-1], [-20, 0]))
               in snake.pos) or
              snake.pos[-1][0] - 20 < 20))
            or (snake.x_ch == -20 and
                snake.y_ch == 0 and
                ((list(map(add, snake.pos[-1], [0, -20]))
                  in snake.pos) or
                 snake.pos[-1][-1] - 20 < 20))
            or (snake.x_ch == 20 and
                snake.y_ch == 0 and
                ((list(map(add, snake.pos[-1], [0, 20]))
                  in snake.pos) or
                 snake.pos[-1][-1] + 20 >=
                 (game.game_h - 20))),

            # danger left
            (snake.x_ch == 0 and
             snake.y_ch == 20 and
             ((list(map(add, snake.pos[-1], [20, 0]))
               in snake.pos) or
              snake.pos[-1][0] + 20 > (game.game_w - 20)))
            or (snake.x_ch == 0 and
                snake.y_ch == -20 and
                ((list(map(add, snake.pos[-1], [-20, 0]))
                  in snake.pos) or snake.pos[-1][0] - 20 < 20))
            or (snake.x_ch == 20 and
                snake.y_ch == 0 and
                ((list(map(add, snake.pos[-1], [0, -20]))
                  in snake.pos) or snake.pos[-1][-1] - 20 < 20))
            or (snake.x_ch == -20 and
                snake.y_ch == 0 and
                ((list(map(add, snake.pos[-1], [0, 20]))
                  in snake.pos) or snake.pos[-1][-1] + 20 >=
                 (game.game_h - 20))),

            # move left     move right      move up           move down
            snake.x_ch == -20, snake.x_ch == 20, snake.y_ch == -20, snake.y_ch == 20,

            # food left     food right          food up             food down
            food.x_f < snake.x, food.x_f > snake.x, food.y_f < snake.y, food.y_f > snake.y
        ]

        for x in range(len(st)):
            if st[x]:
                st[x] = 1
            else:
                st[x] = 0

        return np.asarray(st)

    # set reward
    def set_r(self, snake, cr):
        self.r = 0
        # crashing = punishment -10 points
        if cr:
            self.r = -10
            return self.r
        # eating apple = reward 20 points
        if snake.eaten:
            self.r = 20
        return self.r

    # memo of state action reward etc.
    def rem(self, st, act, r, n_st, done):
        self.memory.append((st, act, r, n_st, done))

    # replaying new for replaying the memory
    def replay_curr(self, memory, batch_size):
        if len(memory) > batch_size:
            batch_m = random.sample(memory, batch_size)
        else:
            # small batch
            batch_m = memory
        for st, act, r, n_st, done in batch_m:
            targ = r
            if not done:
                targ = r + self.gamma * np.amax(self.model.predict(np.array([n_st]))[0])
            t_f = self.model.predict(np.array([st]))
            t_f[0][np.argmax(act)] = targ
            # saves in model
            self.model.fit(np.array([st]), t_f, epochs=1, verbose=0)

    # training short term memory SARSA algorithm based (State action reward state action)
    # makes state  and the action taken by agent results in reward and gets back to the same state for the next action
    def short_mem_train(self, st, act, r, n_st, done):
        targ = r
        # take action, give reward, get state
        if not done:
            targ = r + self.gamma * np.amax(self.model.predict(n_st.reshape((1, 11)))[0])
        t_f = self.model.predict(st.reshape((1, 11)))
        t_f[0][np.argmax(act)] = targ
        # saves in model
        self.model.fit(st.reshape((1, 11)), t_f, epochs=1, verbose=0)
