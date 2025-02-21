import gym
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def discretize_state(state, bin_edges):
    """Discretize a given continuous state into pre-defined bins."""
    return tuple(np.digitize(state[i], bin_edges[i]) - 1 for i in range(6))

def state_to_index(discrete_state, num_bins):
    """Convert a discrete state tuple into a unique index."""
    index = 0
    for i, s in enumerate(discrete_state):
        index *= num_bins[i]
        index += s  # Convert tuple to a single index
    return index

# Create the acrobot gym environment.
env = gym.make("Acrobot-v1")
env.action_space.seed(42)

# Define min and max values for each state.
state_mins = np.array([-1, -1, -1, -1, -4*np.pi, -9*np.pi])
state_maxs = np.array([ 1,  1,  1,  1,  4*np.pi,  9*np.pi]) + 1e-6

# Discretize the state space into bins. 
num_bins = [50] * 6
bin_edges = [np.linspace(state_mins[i], state_maxs[i], num_bins[i] + 1) for i in range(6)]

# Set a random initial state. 
state_ii, info = env.reset()
state_0 = state_ii

# Set parameters for TD learning.
S = np.prod(num_bins) # total number of states (finite)
lambdas = [0.0, 0.3, 0.5, 0.7, 1]
episodes = 40 # number of episodes
N = 4000 # number of samples
alpha = 0.001
gamma = 0.99

# Allocate memory for TD learning.  
tdiff = np.zeros([len(lambdas), N, episodes])

# Loop over multiple lambda values. 
for lam_idx in range(len(lambdas)):

    # Set the new lambda value. 
    lam = lambdas[lam_idx]

    # For each lambda value, run a set of episodes.
    for episode in range(episodes):

        # Set a random initial state. 
        state_ii, info = env.reset()

        # Reset the eligibility trace.
        z = defaultdict(float) 

        # Reset the value function. 
        V = defaultdict(float) 

        # For each episode, take N steps forward in the sim. 
        for k in range(N):

            # Sample a random action.
            action = np.random.choice([0,1,2], p=[0.1,0.7,0.2])
            
            # Apply the action to the environment and get feedback. 
            state_iip1, reward, terminated, truncated, info = env.step(action)

            # Discretize the current/next states.
            disc_state_ii = discretize_state(state_ii, bin_edges)
            disc_state_iip1 = discretize_state(state_iip1, bin_edges)

            # Get the indices of the value func vector. 
            ii = state_to_index(disc_state_ii, num_bins)
            iip1 = state_to_index(disc_state_iip1, num_bins)

            # Update the trace vector.
            # z = lam*gamma*z
            # z[ii] = z[ii] + 1

            # Update the trace vector. 
            for key in z:
                z[key] *= lam * gamma
            z[ii] += 1  # Increment eligibility of current state

            # Update the temporal difference
            td_error = reward + gamma*V[iip1] - V[ii]
            tdiff[lam_idx, k, episode] = td_error

            # Update the value function using TD learning 
            # V[ii] = V[ii] +  alpha*tdiff[ii, episode]
            # V = V +  alpha*td_error*z

            # Update value function using TD learning
            for key in z:
                V[key] += alpha * td_error * z[key]
                # print("Key in z:     ", z[key], key)
                # print("V:            ", V[key], key)

            # Set the new current state = next state
            state_ii = state_iip1

            # Debug
            # print("Current state index: ", ii)
            # print("Next state index:    ", iip1)
            # print("Step: ", k)
            # if k == 5:
            #     exit()
            # input("enter")

        print(f'lambda = {lam}, episode = {episode}')
    
    print('---')

# Shut down the environment.
env.close()

# Compute the tdiff for each sample of each lambda.
# Each row of tdiff_avg are the tdiffs for each sample. 
tdiff_avg = np.sqrt(np.mean(np.square(tdiff), axis=2))  # np array (# lams, N)

# Plot the evolution of tdiff over sample for each lambda.
for ii in range(5):
    plt.plot(tdiff_avg[ii,:], label=f'\u03BB {lambdas[ii]}')
plt.title("Average Temporal Differences Squared Over Sim Steps")
plt.xlabel("Simulation Step Number")
plt.ylabel("Temporal Difference Errors Averaged over Episodes Squared")
plt.legend()
plt.show()