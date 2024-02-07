import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the XOR problem dataset
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# input -> layer_1 -> layer_2 -> output

# Initialize the weights randomly
np.random.seed(42)
weights_ih = np.random.random((2, 2)) 
weights_ho = np.random.random((2, 1)) 

lr = 5e-1

x = x
layer_ih = sigmoid(np.dot(x, weights_ih))
layer_ho = sigmoid(np.dot(layer_ih, weights_ho))
print("before training")
print(layer_ho)

# Train the neural network using backpropagation
for i in range(10000):
    # Forward propagation
    layer_ih = np.dot(x, weights_ih) # x(4,2) @ wih(2,2) -> (4,2)
    sig_ih = sigmoid(layer_ih) # sigmoid(x @ wih) -> (4,2)
    layer_ho = np.dot(sig_ih, weights_ho) # sigmoid(x @ wih)(4,2) @ who(2,1) -> (4,1)
    sig_ho = sigmoid(layer_ho) # sigmoid(sigmoid(x @ wih)(4,2) @ who(2,1)) -> (4,1)
    loss = (y - sig_ho)**2 # (4,1)

    # Backpropagation - inverse direction of forward propagation
    
    # dloss/dsigmoid_ho = -2(y - sigmoid(who * sigmoid(wih * x))) = -2(y - sigmoid_ho)
        # dsigomid_ho/dlayer_ho = sigmoid_derivative(who * sigmoid(wih * x)) = sigmoid_derivative(layer_ho)
            # dlayer_ho/dw_ho = sigmoid(wih * x) = sigmoid_ih = sigmoid(layer_ih)
            # (dlayer_ho/db_ho = 1)
            # dlayer_ho/dsigmoid_ih = weights_ho
                # dsigmoid_ih/dlayer_ih = sigmoid_derivative(wih * x)
                    # dlayer_ih/dw_ih = x

    
    # Fill the bellows
    dloss_dsig_ho = 
    dsig_ho_dlayer_ho =  
    dlayer_ho_dw_ho = 
    dlayer_ho_dsig_ih = 
    dsig_ih_dlayer_ih = 
    dlayer_ih_dw_ih = 
    
    dloss_dlayer_ho = # (4, 1)
    dloss_dsigmoid_ih =  # (4, 2)
    dloss_dlayer_ih = # (4, 2)
        
    dloss_dweights_ho = # (2, 1)
    dloss_dweights_ih = # (2, 2)
    # Fill the aboves

    # Update the weights
    weights_ho += lr * dloss_dweights_ho
    weights_ih += lr * dloss_dweights_ih 

# Test the neural network
x = x
layer_ih = sigmoid(np.dot(x, weights_ih))
layer_ho = sigmoid(np.dot(layer_ih, weights_ho))

print("Predictions:")
print(layer_ho)
