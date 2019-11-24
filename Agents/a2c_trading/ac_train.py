import gym
import numpy as np
import gym_trading
from Agents.a2c_trading.ac_agent import ACAgent
from tools.plot_tool import plot_with_avg_std
env = gym.make('hwenv-v0')

ob_dim = env.observation_space.shape[0]
ac_dim = env.action_space.n

agent = ACAgent(ob_dim, ac_dim)

avg_rews = []
n_iter = 200
for itr in range(n_iter):
    print("********** Iteration %i ************" % itr)
    paths, timesteps_this_batch, avg_rew, avg_info = agent.sample_trajectories(itr, env, info_name='shortfall')
    avg_rews.append(avg_rew)
    print(avg_rew, avg_info)

    ob_no = np.concatenate([path["observation"] for path in paths])
    ac_na = np.concatenate([path["action"] for path in paths])
    re_n = np.concatenate([path["reward"] for path in paths])
    next_ob_no = np.concatenate([path["next_observation"] for path in paths])
    terminal_n = np.concatenate([path["terminal"] for path in paths])

    agent.update_critic(ob_no, next_ob_no, re_n, terminal_n)
    adv_n = agent.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)
    agent.update_actor(ob_no, ac_na, adv_n)

plot_with_avg_std(avg_rews, 1, xlabel=f'Number of Episodes in {1}')