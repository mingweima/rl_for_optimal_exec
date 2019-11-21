import gym
import keras
from CartPoleAgents.a2c_rnn_cartpole import ACRnnAgent
from tools.plot_tool import plot_with_avg_std
env = gym.make('CartPole-v1')

discrete = isinstance(env.action_space, gym.spaces.Discrete)
ob_dim = env.observation_space.shape[0]
ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

agent = ACRnnAgent(ob_dim,
                   ac_dim,
                   batch_size=100)
agent.actor_model = keras.models.load_model('dra2c_cartpole_actor_70_itr.h5')
agent.critic_model = keras.models.load_model('dra2c_cartpole_critic_70_itr.h5')

avg_rews = []
n_iter = 100
total_timesteps = 0
for itr in range(n_iter):
    print("********** Iteration %i ************" % itr)
    ob_seq, next_ob_seq, ac_na, re_n, terminal_n, avg_rew = agent.sample_trajectories(env, render=True, animate_eps_frequency=1)
    avg_rews.append(avg_rew)
    print(avg_rew)

plot_with_avg_std(avg_rews, 1, xlabel=f'Number of Iterations')