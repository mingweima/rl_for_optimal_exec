import numpy as np

class AlmgrenChrissAgent:
    """
    The AlmgrenChriss Agent relies on the Almgren Chriss model to make trading decisions.

        Attributes:
            time_horizon: the number of time steps to fully liquidate the position
            time (int32): current time step
            eta (float64): temporary price impact parameter
            rho (float64): permanant price impact parameter
            sigma (float64): volatility of the stock
            tau (float64): length of discrete time period
            lamb (float64): level of risk aversion (zero if the trader is risk-neutral)
    """
    def __init__(self, time_horizon, eta=2.5e-6, rho=0, sigma=5e-4, tau=60, lamb=0.01):
        self.eta = eta
        self.rho = rho
        self.sigma = sigma
        self.tau = tau
        self.time_horizon, self.time = time_horizon.seconds, 1
        k_bar = np.sqrt(lamb * sigma**2 / (eta * (1 - rho * tau / (2 * eta))))
        self.kappa = (1/tau) * np.arccosh(tau**2 * k_bar**2 * 0.5 + 1)

    def reset(self):
        """
        Reset the current time to 1.
        """
        self.time = 1

    def act(self, inventory):
        """
        Take an action based on the Almgren Chriss Algorithm.
            Args:
                inventory (float64): the ratio of the current position to the initial inventory.
            Returns:
                nj (float64): the ratio of the position to sell at this time step to the current position
        """
        if self.kappa == 0:
            nj = self.tau / self.time_horizon * (1 / inventory)
        else:
            nj = 2 * np.sinh(0.5 * self.kappa * self.tau) * np.cosh(self.kappa * (
                    self.time_horizon - (self.time - 0.5) * self.tau)) / np.sinh(self.kappa * self.time_horizon)
        self.time += 1
        if self.time == self.time_horizon + 1:
            nj = 1
        return nj
