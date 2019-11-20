import numpy as np
import keras
from collections import deque
import random


class DQNAgent(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=int(1e4))
        self.gamma = 1  # discount rate

        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.005
        self.epsilon_decay = 0.995
        self.learning_rate = 1e-4
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = keras.Sequential()
        model.add(keras.layers.Dense(48, input_dim=self.state_size, activation='relu'))
        model.add(keras.layers.Dense(24, activation='relu'))
        model.add(keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Ouput: action signal (from 0 to 9)
        reward_list = self.model.predict(state)[0]
        s = np.exp(reward_list)
        # print(s)
        probability_list = s/np.sum(s)
        prob_mass = 0
        rand = random.random()
        for i in range(10):
            if prob_mass <= rand < prob_mass + probability_list[i]:
                return i
            else:
                prob_mass += probability_list[i]

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)
        minibatch = random.sample(self.memory, batch_size)
        # print(minibatch)
        states = []
        targets = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                         np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            # states.append(state[0])
            # targets.append(target_f[0])
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        # if len(self.memory) < batch_size:
        #     batch_size = len(self.memory)
        # minibatch = random.sample(self.memory, batch_size)
        # for state, action, reward, next_state, done in minibatch:
        #     target = reward
        #     if not done:
        #         target = reward + self.gamma * \
        #                  np.amax(self.model.predict(next_state)[0])
        #     target_f = self.model.predict(state)
        #     target_f[0][action] = target
        #     self.model.fit(state, target_f, epochs=1, verbose=0)
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay
