import gym
from CartPoleAgents.a2c_rnn_cartpole import ACRnnAgent
from tools.plot_tool import plot_with_avg_std
env = gym.make('CartPole-v1')

discrete = isinstance(env.action_space, gym.spaces.Discrete)
ob_dim = env.observation_space.shape[0]
ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

agent = ACRnnAgent(ob_dim,
                   ac_dim,
                   batch_size=2000,
                   learning_rate=1e-2)

avg_rews = []
n_iter = 80
for itr in range(n_iter):
    print("********** Iteration %i ************" % itr)
    ob_seq, next_ob_seq, ac_na, re_n, terminal_n, avg_rew = \
        agent.sample_trajectories(env,
                                  render=False,
                                  animate_eps_frequency=10,
                                  )
    avg_rews.append(avg_rew)
    print(avg_rew)

    agent.update_critic(ob_seq, next_ob_seq, re_n, terminal_n)
    adv_n = agent.estimate_advantage(ob_seq, next_ob_seq, re_n, terminal_n)
    agent.update_actor(ob_seq, ac_na, adv_n)
    if itr == 75:
        agent.actor_model.save(f'dra2c_cartpole_actor_{int(itr)}_itr.h5')
        agent.critic_model.save(f'dra2c_cartpole_critic_{int(itr)}_itr.h5')

plot_with_avg_std(avg_rews, 1, xlabel=f'Number of Episodes in {1}')