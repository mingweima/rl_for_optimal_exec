import matplotlib.pyplot as plt
import numpy as np
import gym
import gym_trading


from A2C.ddpg_agent import Agent


env = gym.make('hwenv-v0')
env.reset()

agent = Agent(state_size=4, action_size=1, random_seed=0, name='lgq')
rewards, episode_list = [], []


round_to_print = 20
for episode in range(1000):
    total_reward = 0
    if episode == 0 or episode % round_to_print == 0:
        print(f'=========================================Episode {episode}===============================================')
    init_obs = np.asarray(env.reset())
    current_obs = np.reshape(env.reset(), [1, 4])
    total_reward = 0
    for time_t in range(31):
        state = init_obs
        action = agent.act(state)
        # print(action)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, 4])
        total_reward += reward
        agent.step(state, action, reward, next_state, done)
        state = next_state
        if done:
            if episode % round_to_print == 0:
                print('episode: {}/{}, total reward: {}, total time: {}'.format(episode, 1000, total_reward, time_t))
            rewards.append(total_reward)
            episode_list.append(episode)
            break

plt.plot(episode_list, rewards)
plt.show()
