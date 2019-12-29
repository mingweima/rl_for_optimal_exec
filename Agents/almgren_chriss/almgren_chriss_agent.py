from copy import deepcopy
import numpy as np


class AlmgrenChrissAgent:
    def __init__(self, env, time_horizon, eta, rho, sigma, tau, lamb, kappa):
        self.ac_dict = deepcopy(env.ac_dict)
        self.ac_type = deepcopy(env.ac_type)
        self.eta = eta
        self.rho = rho
        self.sigma = sigma
        self.tau = tau
        self.time_horizon = time_horizon
        self.steps, self.j = time_horizon / self.tau, 1
        # k_bar = np.sqrt(abs(lamb * sigma**2 / (eta * (1 - rho * tau / (2 * eta)))))
        # self.kappa = (1/tau) * np.arccosh(tau**2 * k_bar**2 * 0.5 + 1)
        self.kappa = kappa

    def reset(self):
        self.j = 1

    def act(self, state):

        def closest_action(nj):
            action = 0
            difference = abs(self.ac_dict[action] - nj)
            for ac, proportion in self.ac_dict.items():
                if (proportion - nj) < difference:
                    action = ac
                    difference = abs(self.ac_dict[action] - nj)
            return action

        if self.ac_type == 'vanilla_action':
            inventory = state[0][1]
            if self.kappa == 0:
                nj = self.tau / self.time_horizon * (1 / inventory)
            else:
                nj = 2 * np.sinh(0.5 * self.kappa * self.tau) * np.cosh(self.kappa * (
                        self.time_horizon - (self.j - 0.5) * self.tau)) * (1 / inventory) / np.sinh(self.kappa
                                                                                                    * self.time_horizon)
            self.j += 1
            if self.j == self.steps + 1:
                nj = 1
            action = closest_action(nj)

            return action
        elif self.ac_type == 'prop_of_ac':
            action = closest_action(1)
            return action
        else:
            raise Exception('Unknown Action Type')
