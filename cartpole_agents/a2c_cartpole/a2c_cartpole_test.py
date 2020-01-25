import gym
import keras
from cartpole_agents.a2c_cartpole.ac_agent import ACAgent
from tools.plot_tool import plot_with_avg_std
env = gym.make('CartPole-v1')

discrete = isinstance(env.action_space, gym.spaces.Discrete)
ob_dim = env.observation_space.shape[0]
ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

agent = ACAgent(ob_dim, ac_dim)
agent.actor_model = keras.models.load_model('a2c_cartpole_actor_50_eps.h5')
agent.critic_model = keras.models.load_model('a2c_cartpole_critic_50_eps.h5')


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
    paths, timesteps_this_batch, avg_rew = agent.sample_trajectories(env, render=True, animate_eps_frequency=1)
    avg_rews.append(avg_rew)
    print(avg_rew)
    total_timesteps += timesteps_this_batch

plot_with_avg_std(avg_rews, 1, xlabel=f'Number of Episodes in {1}')