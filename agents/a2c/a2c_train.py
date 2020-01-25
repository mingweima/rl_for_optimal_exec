import numpy as np
import gym
import gym_trading_deprecated

from agents.a2c.a2c_agent import A2CAgent
from tools.plot_tool import plot_with_avg_std

def A2CTrain(scenario_args, observation_space_args,
             action_space_args, reward_args, data_args, almgren_chriss_args):
    """
    Train the A2CAgent by sampling trajectories from the trading_environment.
    """
    N_ITERATION = 200

    # Initialize the gym trading_environment.
    env = gym.make('hwenv-v0',
                   scenario_args=scenario_args,
                   observation_space_args=observation_space_args,
                   action_space_args=action_space_args,
                   reward_args=reward_args,
                   data_args=data_args,
                   almgren_chriss_args=almgren_chriss_args)

    # Initialize the A2CAgent.
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n
    agent = A2CAgent(ob_dim, ac_dim)

    # Run the iterations through recursively sampling trajectories and update neural network parameters.
    avg_rews = []
    for itr in range(N_ITERATION):
        # Sample trajectories
        print("********** Iteration %i ************" % itr)
        paths, timesteps_this_batch, avg_rew, avg_info = agent.sample_trajectories(itr, env, info_name='shortfall')
        avg_rews.append(avg_rew)
        print("Total rewards per trajectory in this iteration: ", avg_rew)
        ob_no = np.concatenate([path["observation"] for path in paths])
        ac_na = np.concatenate([path["action"] for path in paths])
        re_n = np.concatenate([path["reward"] for path in paths])
        next_ob_no = np.concatenate([path["next_observation"] for path in paths])
        terminal_n = np.concatenate([path["terminal"] for path in paths])

        # Update the critic model and the actor model
        agent.update_critic(ob_no, next_ob_no, re_n, terminal_n)
        adv_n = agent.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)
        agent.update_actor(ob_no, ac_na, adv_n)

    # Visualize the training results.
    plot_with_avg_std(avg_rews, 1, xlabel=f'Number of Episodes in {1}')