import gym
import gym_trading
import numpy as np
from Agents.dpg_trading.pg_agent import PGAgent
from tools.plot_tool import plot_with_avg_std
env = gym.make('hwenv-v0')

discrete = isinstance(env.action_space, gym.spaces.Discrete)
ob_dim = env.observation_space.shape[0]
ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

agent = PGAgent(ob_dim, ac_dim)
n_iter = 100
avg_rews = []
for itr in range(n_iter):
    print("******************** Iteration %i **********************" % itr)
    paths, timesteps_this_batch, avg_rew, avg_info = agent.sample_trajectories(itr, env, info_name='shortfall')
    avg_rews.append(avg_rew)
    print(f'Average reward: {avg_rew}, Average shortfall: {avg_info}.')
    ob_no = np.concatenate([path["observation"] for path in paths])
    ac_na = np.concatenate([path["action"] for path in paths])
    re_n = [path["reward"] for path in paths]


    adv_n = agent.estimate_return(ob_no, re_n)
    agent.update_parameters(ob_no, ac_na, adv_n)


plot_with_avg_std(avg_rews, 1, xlabel=f'Number of Episodes in {1}')
