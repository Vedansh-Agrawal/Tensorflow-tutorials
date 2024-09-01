import gym
import numpy as np
import time

env = gym.make('FrozenLake-v1', render_mode="rgb_array")

STATES = env.observation_space.n
ACTIONS = env.action_space.n

Q = np.zeros((STATES, ACTIONS))

EPISODES = 1500
MAX_STEPS = 100

LEARNING_RATE = 0.81
GAMMA = 0.96

epsilon = 0.9


rewards = []
for episodes in range(EPISODES):
    state = env.reset()

    for _ in range(MAX_STEPS):

        env.render()

        if np.random.uniform(0,1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])

        new_state, reward, done, _ = env.step(action)

        Q[state, action] = Q[state, action] + LEARNING_RATE*(reward + GAMMA*np.max(Q[new_state, :] - Q[state, action]))

        state = new_state

        if done:
            rewards.append(reward)
            epsilon -= 0.001
            break

