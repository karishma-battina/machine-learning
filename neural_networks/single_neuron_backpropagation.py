# This code demonstrates a simple gradient descent optimization process for a single neuron with a ReLU activation function.
# The goal is to adjust the weights and bias to minimize the squared error between the predicted output and a target output (0.0).

import numpy as np

# Initial parameters
weights = np.array([-3.0, -1.0, 2.0])
bias = 1.0
inputs = np.array([1.0, -2.0, 3.0])
target_output = 0.0
learning_rate = 0.001

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0)

for iteration in range(200):
    # Forward pass
    linear_output = np.dot(weights, inputs) + bias  # Computes the weighted sum of inputs and adds bias.
    output = relu(linear_output)
    loss = (output - target_output) ** 2   # Squared error loss between predicted and target output.

    # Backward pass
    dloss_doutput = 2 * (output - target_output)   # Derivative of loss with respect to the output.
    doutput_dlinear = relu_derivative(linear_output)   # Returns 1 if linear_output > 0, else 0.
    dlinear_dweights = inputs
    dlinear_dbias = 1.0

    dloss_dlinear = dloss_doutput * doutput_dlinear   # chain rule
    dloss_dweights = dloss_dlinear * dlinear_dweights   # gradient for weights
    dloss_dbias = dloss_dlinear * dlinear_dbias    # gradient for bias

    # Update weights and bias
    weights -= learning_rate * dloss_dweights
    bias -= learning_rate * dloss_dbias

    # Print the loss for this iteration
    print(f"Iteration {iteration + 1}, Loss: {loss}")

print("Final weights:", weights)
print("Final bias:", bias)

# ReLU Behavior: If linear_output <= 0, gradients become 0, and no updates occur (dead neuron problem).
# Here, the initial linear_output is 6 (positive), so gradients flow initially.

# The loss starts high (e.g., 36 at iteration 1) and decreases as weights/bias are adjusted.
# Eventually, the network learns to output 0 (matching the target), and loss drops to 0.

# The final weights and bias will stabilize when the network predicts 0 consistently.

# It’s a minimal example of how neural networks adjust parameters during training.
# In practice, we’d use frameworks like TensorFlow/PyTorch instead of manual gradient calculations.