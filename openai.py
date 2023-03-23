import gym
import numpy as np

# Initialize environment and algorithm parameters
env = gym.make('CartPole-v0')
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n
learning_rate = 0.1
discount_factor = 0.99
epsilon = 0.1
q_table = np.zeros((n_states, n_actions))

# Define policy function for selecting actions
def epsilon_greedy_policy(state):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])

# Run algorithm for some number of episodes
n_episodes = 10000
for episode in range(n_episodes):
    state = env.reset()
    done = False
    while not done:
        action = epsilon_greedy_policy(state)
        next_state, reward, done, info = env.step(action)
        q_value = q_table[state][action]
        next_max_q_value = np.max(q_table[next_state])
        td_error = reward + discount_factor * next_max_q_value - q_value
        q_table[state][action] += learning_rate * td_error
        state = next_state