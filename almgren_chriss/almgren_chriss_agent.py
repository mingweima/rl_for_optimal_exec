import numpy as np
from collections import deque
import random


class AlmgrenChrissAgent:
    def __init__(self, time_horizon=60, eta=2.5e-6, rho=0, sigma=1e-3, tau=1, lamb=0.1):
        self.eta = eta
        self.rho = rho
        self.sigma = 1.8e-4
        self.tau = 1
        self.T, self.j = time_horizon, 1
        k_bar = np.sqrt(lamb * sigma**2 / (eta * (1 - rho * tau / (2 * eta))))
        self.kappa = (1/tau) * np.arccosh(tau**2 * k_bar**2 * 0.5 + 1)

    def remember(self, state, action, reward, next_state, done):
        pass

    def reset(self):
        self.j = 1

    def act(self, state):
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
        return nj

    def replay(self, batch_size):
        pass