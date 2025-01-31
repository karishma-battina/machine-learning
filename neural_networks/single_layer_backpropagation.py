# Inputs: A 1D array [1, 2, 3, 4] representing input features.
# Weights: A 3x4 matrix connecting 4 inputs to 3 neurons.
# Biases: A 1D array [0.1, 0.2, 0.3] for the 3 neurons.
# Learning Rate: 0.001, controlling the step size during gradient descent.
# ReLU Activation: Used to introduce non-linearity.
import numpy as np

# Initial inputs
inputs = np.array([1, 2, 3, 4])

# Initial weights and biases
weights = np.array([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.9, 1.0, 1.1, 1.2]
])

biases = np.array([0.1, 0.2, 0.3])

# Learning rate
learning_rate = 0.001

# ReLU activation function and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Training loop
for iteration in range(200):
    # Forward pass. Computes a weighted sum of inputs for each neuron. Shape of z: (3,) (3 neurons).
    z = np.dot(weights, inputs) + biases
    # Applies ReLU to z, turning negative values to 0.
    a = relu(z)
    # Sums all activated values in a to produce a scalar output.
    y = np.sum(a)

    # Calculate loss. The goal is to minimize the squared sum of activations (y).
    loss = y ** 2

    # Backward pass
    # Gradient of loss with respect to output y. Derivative of loss = y² is 2y.
    dL_dy = 2 * y

    # Gradient of y with respect to a. Since y is the sum of a, each element in a contributes 1 to the gradient.
    dy_da = np.ones_like(a)

    # Gradient of loss with respect to a
    dL_da = dL_dy * dy_da

    # Gradient of a with respect to z (ReLU derivative). ReLU derivative is 1 where z > 0, 0 otherwise.
    da_dz = relu_derivative(z)

    # Gradient of loss with respect to z. Combines gradients from y and ReLU.
    dL_dz = dL_da * da_dz

    # Gradient of z with respect to weights and biases
    dL_dW = np.outer(dL_dz, inputs)
    dL_db = dL_dz

    # Update weights and biases
    weights -= learning_rate * dL_dW
    biases -= learning_rate * dL_db

    # Print the loss every 20 iterations
    if iteration % 20 == 0:
        print(f"Iteration {iteration}, Loss: {loss}")

# Final weights and biases
print("Final weights:\n", weights)
print("Final biases:\n", biases)


# Goal: The network learns to minimize the sum of ReLU outputs (y), effectively pushing activations toward zero.
# ReLU Behavior: Neurons with z ≤ 0 become "dead" (gradient = 0) and stop updating.
# Final Output: After training, weights/biases will be adjusted such that a ≈ 0 (to minimize y).
# This code demonstrates manual backpropagation without frameworks like TensorFlow. Shows how gradients flow through ReLU and linear layers.
# Useful for understanding low-level neural network mechanics.
