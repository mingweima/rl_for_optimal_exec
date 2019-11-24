import numpy as np
from collections import deque
import random
from copy import deepcopy

class AlmgrenChrissAgent:
    def __init__(self, env, time_horizon=60, eta=2.5e-6, rho=0, sigma=1e-3, tau=1, lamb=0):
        self.ac_num_to_act_dict = deepcopy(env.ac_num_to_act_dict)
        self.eta = eta
        self.rho = rho
        self.sigma = sigma
        self.tau = tau
        self.T, self.j = time_horizon, 1
        k_bar = np.sqrt(lamb * sigma**2 / (eta * (1 - rho * tau / (2 * eta))))
        self.kappa = (1/tau) * np.arccosh(tau**2 * k_bar**2 * 0.5 + 1)

    def remember(self, state, action, reward, next_state, done):
        pass

    def reset(self):
        self.j = 1

    def act(self, state):
        def closest_action(nj):
            action = 0
            difference = abs(self.ac_num_to_act_dict[action] - nj)
            for ac, proportion in self.ac_num_to_act_dict.items():
                if (proportion - nj) < difference:
                    action = ac
            return action

        inventory = state[0][1]
        if self.kappa == 0:
            nj = self.tau / self.T * (1 / inventory)
        else:
            nj = 2 * np.sinh(0.5 * self.kappa * self.tau) * np.cosh(self.kappa * (
                self.T - (self.j - 0.5) * self.tau)) * (1 / inventory) / np.sinh(self.kappa * self.T)
            # nj = 2 * np.sinh(0.5 * self.kappa * self.tau) * np.cosh(self.kappa * (
            #         self.T - (self.j - 0.5) * self.tau)) / np.sinh(self.kappa * self.T)
        self.j += 1
        if self.j == self.T + 1:
            nj = 1
        action = closest_action(nj)

        return action


    def replay(self, batch_size):
        pass