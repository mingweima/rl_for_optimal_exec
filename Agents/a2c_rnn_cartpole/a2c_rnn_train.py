import gym
import numpy as np
from Agents.a2c_rnn_cartpole.a2c_rnn_agent import ACRnnAgent
from tools.plot_tool import plot_with_avg_std
env = gym.make('CartPole-v1')

discrete = isinstance(env.action_space, gym.spaces.Discrete)
ob_dim = env.observation_space.shape[0]
ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

agent = ACRnnAgent(ob_dim, ac_dim)


# # build computation graph
# agent.build_computation_graph()
#
# # tensorflow: config, session, variable initialization
# agent.init_tf_sess()

avg_rews = []
n_iter = 100
for itr in range(n_iter):
    print("********** Iteration %i ************" % itr)
    ob_seq, next_ob_seq, ac_na, re_n, terminal_n, avg_rew = \
        agent.sample_trajectories(env, render=True, animate_eps_frequency=10)
    avg_rews.append(avg_rew)
    print(avg_rew)

    agent.update_critic(ob_seq, next_ob_seq, re_n, terminal_n)
    adv_n = agent.estimate_advantage(ob_seq, next_ob_seq, re_n, terminal_n)
    agent.update_actor(ob_seq, ac_na, adv_n)

plot_with_avg_std(avg_rews, 1, xlabel=f'Number of Episodes in {1}')