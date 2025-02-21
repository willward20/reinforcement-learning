import gym
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def discretize_state(state, bin_edges):
    """Discretize a given continuous state into pre-defined bins."""
    return tuple(np.digitize(state[i], bin_edges[i]) - 1 for i in range(4))

def state_to_index(discrete_state, num_bins):
    """Convert a discrete state tuple into a unique index."""
    index = 0
    for i, s in enumerate(discrete_state):
        index *= num_bins[i]
        index += s  # Convert tuple to a single index
    return index

# Create the acrobot gym environment.
env = gym.make("CartPole-v1")#, render_mode="human")
env.action_space.seed(42)

# Define min and max values for each state.
state_mins = np.array([-4.8, -5, -0.418, -5])
state_maxs = np.array([ 4.8,  5,  0.418,  5]) + 1e-6

# Discretize the state space into bins. 
num_bins = [50] * 4
bin_edges = [np.linspace(state_mins[i], state_maxs[i], num_bins[i] + 1) for i in range(4)]

# Set a random initial state. 
state_ii, info = env.reset()

# Set parameters for TD learning.
S = np.prod(num_bins) # total number of states (finite)
A = 2 # total number of actions (push left or push right)
episodes = 750 # number of episodes
N = 2000 # number of samples
alpha = 0.1
gamma = 0.9

# Initialize the Q-table. 
Q = np.zeros([S, A]) # entries for each (state, action) pair
Q_next = Q

cum_rewards = []

# Choose epsilon for exploit/explore ratio
eps = 0.9

# Update the Q-table over each episode.
for episode in range(episodes):

    # Set a random initial state. 
    state_ii, info = env.reset()

    # Get the index of the initial state. 
    disc_state_ii = discretize_state(state_ii, bin_edges)
    ii = state_to_index(disc_state_ii, num_bins)

    # Track cumulative reward over each episode.
    cum_reward = 0.0
    cum_rewards_episode = []

    # For each episode, take N steps forward in the sim. 
    for k in range(N):

        # If this is the first episode, sample a random action.
        if episode == 0:
            action = np.random.choice([0, 1], p=[0.5, 0.5])
        # If the Q values are the same, sample a random action.
        elif Q[ii, 0] == 0 and Q[ii, 1] == 0:
            action = np.random.choice([0, 1], p=[0.5, 0.5])
        # Otherwise, take the best action. 
        else:
            if Q[ii, 0] > Q[ii, 1]:
                action = np.random.choice([0, 1], p=[eps, 1-eps])
            else:
                action = np.random.choice([0, 1], p=[1-eps, eps])
        
        # Apply the action to the environment and get feedback. 
        state_iip1, reward, terminated, truncated, info = env.step(action)

        # Accumulate reward
        cum_reward += reward

        # Discretize the next state and get its index.
        disc_state_iip1 = discretize_state(state_iip1, bin_edges)
        iip1 = state_to_index(disc_state_iip1, num_bins)

        # Get the action that maximizes the Q-value for state ii+1
        if Q_next[iip1, 0] > Q_next[iip1, 1]:
            Qmax_iip1 = Q_next[iip1, 0]
        else:
            Qmax_iip1 = Q_next[iip1, 1]

        # Update the Q table. 
        Q_next[ii, action] = Q_next[ii, action] + alpha*(reward + gamma*Qmax_iip1 - Q_next[ii, action])

        # Set the new current state. 
        if terminated or truncated:
            state_ii, info = env.reset()
            cum_rewards_episode.append(cum_reward)
            cum_reward = 0.0
        else:
            state_ii = state_iip1

        # Get the index of the initial state for next iteration. 
        disc_state_ii = discretize_state(state_ii, bin_edges)
        ii = state_to_index(disc_state_ii, num_bins)

    # Pass off the newly-learned Q-table to the next episode
    Q = Q_next

    print(f'episode = {episode}, avg_cum_reward = {np.mean(cum_rewards_episode)}')
    cum_rewards.append(np.mean(cum_rewards_episode))


# Shut down the environment.
env.close()

# Plot the accumulated rewards over each episode.
plt.plot(cum_rewards)
plt.legend()
plt.title("Average Reward Accumulated Over Each Episode")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.show()

# Create a new test environment.
env = gym.make("CartPole-v1", render_mode="human")
env.action_space.seed(42)

# Set a random initial state. 
state_ii, info = env.reset()

# Get the index of the initial state. 
disc_state_ii = discretize_state(state_ii, bin_edges)
ii = state_to_index(disc_state_ii, num_bins)

# Track cumulative reward
cum_reward = 0.0

for k in range(10000):

    # If the Q values are the same, sample a random action.
    if Q[ii, 0] == 0 and Q[ii, 1] == 0:
        action = np.random.choice([0, 1], p=[0.5, 0.5])
    # Otherwise, take the best action. 
    else:
        if Q[ii, 0] > Q[ii, 1]:
            action = 0
        else:
            action = 1
    # print("action: ", action)
    
    # Apply the action to the environment and get feedback. 
    state_iip1, reward, terminated, truncated, info = env.step(action)

    # Accumulate reward
    cum_reward += reward

    # Discretize the next state and get its index.
    disc_state_iip1 = discretize_state(state_iip1, bin_edges)
    iip1 = state_to_index(disc_state_iip1, num_bins)

    # Set the new current state. 
    if terminated or truncated:
        state_ii, info = env.reset()
        print("Cumulative reward: ", cum_reward)
        cum_reward = 0.0
    else:
        state_ii = state_iip1

    # Get the index of the initial state for next iteration. 
    disc_state_ii = discretize_state(state_ii, bin_edges)
    ii = state_to_index(disc_state_ii, num_bins)