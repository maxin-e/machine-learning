import gym
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

gym.envs.register(
    id='MountainCarMyEasyVersion-v9',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=100000,  # MountainCar-v0 uses 200
)

env = gym.make('MountainCarMyEasyVersion-v9')

# Initialize hyper parameters
GAMMA = 0.9
ALPHA = 0.2

# Discretize position-velocity space
min_position = -1.2
max_position = 0.6
max_speed = 0.07
goal_position = 0.5
position_bins = np.linspace(min_position, max_position + 0.01, num=40, endpoint=False)
velocities_bins = np.linspace(-max_speed, max_speed + 0.001, num=40, endpoint=False)

# Initialize tables
states = list()
for x in range(0, len(position_bins)):
    for y in range(0, len(velocities_bins)):
        states.append((position_bins[x], velocities_bins[y]))
q_table = np.zeros([len(position_bins), len(velocities_bins), 3])


# Return index of state in discretized space
def get_state_idx(observation):
    idx = (np.digitize(observation[0], position_bins), np.digitize(observation[1], velocities_bins))
    return idx[0] - 1, idx[1] - 1


# Return max q value for this indexes of bins
def get_max_action(idx):
    return np.argmax(q_table[idx[0], idx[1], :])


def q_learning(episodes):
    epsilon = 0.2

    for x in range(1, episodes):
        old_observation = env.reset()  # Return back to the center
        done = False

        while not done:
            old_idx = get_state_idx(old_observation)

            if random.random() > epsilon:  # Exploitation: choose the best action
                action = get_max_action(old_idx)
            else:
                action = env.action_space.sample()  # Exploration: choose random action

            new_observation, reward, done, info = env.step(action)
            new_idx = get_state_idx(new_observation)
            q_table[old_idx + (action,)] = q_table[old_idx + (action,)] + ALPHA \
                                           * (reward + GAMMA * np.amax(q_table[new_idx]) - q_table[old_idx + (action,)])
            old_observation = new_observation


def play():
    observation = env.reset()
    done = False
    while not done:
        env.render()
        action = get_max_action(get_state_idx(observation))
        observation, reward, done, info = env.step(action)


q_learning(200)
play()

env.close()

q_values = np.max(q_table, axis=2)
graph = sns.heatmap(q_values, linewidths=0.1)
plt.show()

plt.savefig('test.png')
