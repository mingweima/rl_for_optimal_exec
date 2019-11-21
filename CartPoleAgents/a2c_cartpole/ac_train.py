import gym
import numpy as np
from CartPoleAgents.a2c_cartpole.ac_agent import ACAgent
from tools.plot_tool import plot_with_avg_std
env = gym.make('CartPole-v1')

discrete = isinstance(env.action_space, gym.spaces.Discrete)
ob_dim = env.observation_space.shape[0]
ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

agent = ACAgent(ob_dim, ac_dim)


# # build computation graph
# agent.build_computation_graph()
#
# # tensorflow: config, session, variable initialization
# agent.init_tf_sess()

avg_rews = []
n_iter = 100
total_timesteps = 0
for itr in range(n_iter):
    print("********** Iteration %i ************" % itr)
    paths, timesteps_this_batch, avg_rew = agent.sample_trajectories(env, render=True, animate_eps_frequency=10)
    avg_rews.append(avg_rew)
    print(avg_rew)
    total_timesteps += timesteps_this_batch

    ob_no = np.concatenate([path["observation"] for path in paths])
    ac_na = np.concatenate([path["action"] for path in paths])
    re_n = np.concatenate([path["reward"] for path in paths])
    next_ob_no = np.concatenate([path["next_observation"] for path in paths])
    terminal_n = np.concatenate([path["terminal"] for path in paths])

    agent.update_critic(ob_no, next_ob_no, re_n, terminal_n)
    adv_n = agent.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)
    agent.update_actor(ob_no, ac_na, adv_n)
    if itr == 50:
        agent.actor_model.save(f'a2c_cartpole_actor_{int(itr)}_eps.h5')
        agent.critic_model.save(f'a2c_cartpole_critic_{int(itr)}_eps.h5')

plot_with_avg_std(avg_rews, 1, xlabel=f'Number of Episodes in {1}')