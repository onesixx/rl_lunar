import numpy as np


# Initial state probabilities
initial_state = np.array([0.6, 0.4])

# Define the transition matrix
# Each row represents the probabilities of transitioning to other states
transition_matrix = np.array([[0.7, 0.3], [0.4, 0.6]])


# Simulate the Markov chain for a few steps
current_state = np.random.choice([0, 1], p=initial_state)
num_steps = 10

print("Initial state:", current_state)

for _ in range(num_steps):
    next_state = np.random.choice([0, 1], p=transition_matrix[current_state])
    print("Next state:", next_state)
    current_state = next_state
