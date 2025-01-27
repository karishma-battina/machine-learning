# Create a neural network from scratch

# 1. Coding a neuron.
# A neuron is a single unit which has inputs, weights and a bias. Each input is associated with a weight.
inputs = [1, 2, 3]
weights = [0.2, 0.8, -0.5] #we can think of weights as assigning importance to the inputs. Higher weight, higher importance of that input.
bias = 2

outputs = (inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + bias)
print(outputs)

#########################

# 2. Layers
# Let's consider a layer of 3 neurons. Each neuron has 4 inputs. 
inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
 [0.5, -0.91, 0.26, -0.5],
 [-0.26, -0.27, 0.17, 0.87]]

weights1 = weights[0] #LIST OF WEIGHTS ASSOCIATED WITH 1ST NEURON : W11, W12, W13, W14
weights2 = weights[1] #LIST OF WEIGHTS ASSOCIATED WITH 2ND NEURON : W21, W22, W23, W24
weights3 = weights[2] #LIST OF WEIGHTS ASSOCIATED WITH 3RD NEURON : W31, W32, W33, W34

biases = [2, 3, 0.5]

bias1 = 2
bias2 = 3
bias3 = 0.5

outputs = [
 # Neuron 1:
 inputs[0]*weights1[0] +
 inputs[1]*weights1[1] +
 inputs[2]*weights1[2] +
 inputs[3]*weights1[3] + bias1,
 # Neuron 2:
 inputs[0]*weights2[0] +
 inputs[1]*weights2[1] +
 inputs[2]*weights2[2] +
 inputs[3]*weights2[3] + bias2,
 # Neuron 3:
 inputs[0]*weights3[0] +
 inputs[1]*weights3[1] +
 inputs[2]*weights3[2] +
 inputs[3]*weights3[3] + bias3]

print(outputs)

################################
#3. Let's use loops instead of summation, for easier coding

inputs = [1, 2, 3, 2.5]

##LIST OF WEIGHTS
weights = [[0.2, 0.8, -0.5, 1],
 [0.5, -0.91, 0.26, -0.5],
 [-0.26, -0.27, 0.17, 0.87]]

##LIST OF BIASES
biases = [2, 3, 0.5]

# Output of current layer
layer_outputs = []

# For each neuron
for neuron_weights, neuron_bias in zip(weights, biases):
 # Zeroed output of given neuron
 neuron_output = 0
 # For each input and weight to the neuron
 for n_input, weight in zip(inputs, neuron_weights):
 # Multiply this input by associated weight
 # and add to the neuron's output variable
   neuron_output += n_input*weight ## W31*X1 + W32*X2 + W33*X3 + W34*X4
   # Add bias
 neuron_output += neuron_bias ## ## W31*X1 + W32*X2 + W33*X3 + W34*X4 + B3
 # Put neuron's result to the layer's output list
 layer_outputs.append(neuron_output)
print(layer_outputs)

##############################
#Basic numpy operations
import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32

A = np.array([[1, 2], [3, 4]])  # Shape (2,2)
B = np.array([[5, 6], [7, 8]])  # Shape (2,2)
result = np.dot(A, B)
# [[1*5 + 2*7, 1*6 + 2*8],
#  [3*5 + 4*7, 3*6 + 4*8]] → [[19, 22], [43, 50]]
print(result)

C = np.random.rand(3, 2, 5)  # Shape (3, 2, 5)
D = np.random.rand(5, 4)     # Shape (5, 4)
result = np.dot(C, D)        # Shape (3, 2, 4)
print(result)

A = [[1, 2, 3], [4, 5, 6], [7, 8,9]]
print(np.sum(A))

##################################
# 4. Coding a neuron with numpy
import numpy as np
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0

# Convert lists to numpy arrays
inputs_array = np.array(inputs)
weights_array = np.array(weights)

# Calculate the dot product and add the bias
outputs = np.dot(weights_array, inputs_array) + bias

print(outputs)

#################################
# 5. Layer of neurons with numpy
import numpy as np

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

# Convert lists to numpy arrays
inputs_array = np.array(inputs)
weights_array = np.array(weights)
biases_array = np.array(biases)

# Calculate the dot product and add the biases
layer_outputs = np.dot(weights_array, inputs_array) + biases_array
print(layer_outputs)

##############################################
#NOTE: np.dot(W,x)+b == np.dot(x, WT)+b
# 6. Taking Transpose of weight matrix
import numpy as np

inputs = [[1.0, 2.0, 3.0, 2.5], 
          [2.0, 5.0, -1.0, 2.0], 
          [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

# Convert lists to numpy arrays
inputs_array = np.array(inputs)
weights_array = np.array(weights)
biases_array = np.array(biases)

# Calculate the dot product and add the biases
outputs = np.dot(inputs_array, weights_array.T) + biases_array
print(outputs)

#########################################
#Till now, we coded a single layer.
#Next, we code a bunch of layers.
# 7. 2 layers and 3 batches of data
import numpy as np

inputs = [[1, 2, 3, 2.5], #batch 1 with 4 inputs
          [2., 5., -1., 2],   #batch 2 with 4 inputs
          [-1.5, 2.7, 3.3, -0.8]]   #batch 3 with 4 inputs

#let's consider an output layer with 3 neurons. So there will be 3 outputs for every batch
#overall 1 input layer with 4 neurons, 1 hidden layer with 3 neurons and an output layer with 3 neurons.

weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

# Convert lists to numpy arrays
inputs_array = np.array(inputs)
weights_array = np.array(weights)
biases_array = np.array(biases)
weights2_array = np.array(weights2)
biases2_array = np.array(biases2)

# Calculate the output of the first layer
layer1_outputs = np.dot(inputs_array, weights_array.T) + biases_array

# Calculate the output of the second layer
layer2_outputs = np.dot(layer1_outputs, weights2_array.T) + biases2_array

print(layer2_outputs)
