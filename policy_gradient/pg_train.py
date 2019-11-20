import gym
import numpy as np
from policy_gradient.pg_keras import PGAgent

env = gym.make('CartPole-v1')

discrete = isinstance(env.action_space, gym.spaces.Discrete)
ob_dim = env.observation_space.shape[0]
ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

agent = PGAgent(ob_dim, ac_dim)
n_iter = 100
total_timesteps = 0
for itr in range(n_iter):
    print("********** Iteration %i ************" % itr)
    paths, timesteps_this_batch = agent.sample_trajectories(itr, env)
    total_timesteps += timesteps_this_batch

    ob_no = np.concatenate([path["observation"] for path in paths])
    ac_na = np.concatenate([path["action"] for path in paths])
    re_n = [path["reward"] for path in paths]

    adv_n = agent.estimate_return(ob_no, re_n)
    agent.update_parameters(ob_no, ac_na, adv_n)



