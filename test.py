import gym
import gym_simulator
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

env = gym.make('simulator-v0')

observation_size = env.observation_space.shape[0]



batch_size = 32  # A tunable hyperparameter.

states = tf.placeholder(tf.float32, shape=(batch_size, observation_size), name='state')
states_next = tf.placeholder(tf.float32, shape=(batch_size, observation_size), name='state_next')
actions = tf.placeholder(tf.int32, shape=(batch_size,), name='action')
rewards = tf.placeholder(tf.float32, shape=(batch_size,), name='reward')
done_flags = tf.placeholder(tf.float32, shape=(batch_size,), name='done')

# Q = np.zeros([env.observation_space.n,env.action_space.n])
# eta = .628
# gma = .9
# epis = 1000
# total_reward_list = [] # rewards per episode calculate

# for i in range(epis):
#     state = env.reset()
#     total_reward = 0 # Sum of rewards of this episode
#     done = False
#     #The Q-Table learning algorithm
#     for i in range(100):
#         action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) * (1/(i + 1)))
#         state_new, reward, done, _ = env.step(action)
#         Q[state, action] = Q[state, action] + eta * (reward +
#                                                      gma * np.max(Q[state_new, :]) - Q[state, action])
#         total_reward += reward
#         state = state_new
#         if done == True:
#             break
#     total_reward_list.append(total_reward)
