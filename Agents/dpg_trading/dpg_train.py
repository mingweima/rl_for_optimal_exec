import numpy as np
import gym
import gym_trading

from Agents.dpg_trading.dpg_agent import DPGAgent
from tools.plot_tool import plot_with_avg_std


def main():
    """
    Train the DPGAgent by sampling trajectories from the environment.
    """

    # Initialize the gym environment.
    env = gym.make('hwenv-v0')

    # Initialize the DPGAgent.
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n
    agent = DPGAgent(ob_dim, ac_dim)

    # Run the iterations through recursively sampling trajectories and update neural network parameters.
    n_iter = 100
    avg_rews = []
    for itr in range(n_iter):
        print("********** Iteration %i ************" % itr)
        paths, timesteps_this_batch, avg_rew, avg_info = agent.sample_trajectories(itr, env, info_name='shortfall')
        avg_rews.append(avg_rew)
        print("Total rewards per trajectory in this iteration: ", avg_rew)
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        re_n = [path["reward"] for path in paths]
        adv_n = agent.estimate_return(re_n)
        agent.update_parameters(ob_no, ac_na, adv_n)

    # Visualize the training results.
    plot_with_avg_std(avg_rews, 1, xlabel=f'Number of Episodes in {1}')

if __name__ == "__main__":
    main()
