import numpy as np

np.random.seed(3)

#############################################################
####################### INSTRUCTIONS ########################
#############################################################
"""
Goal: Implement a simple Neural Network to for binary classification over
2-dimensional input data. The network must have 1 hidden layer containing 4 units.
The network should utilize batch training. The batch size is set in the 
"Parameters" section.

Please fill in the 4 functions in the section labeled "Fill In These Functions."
The function signatures must not change, and must return appropriate outputs based 
on the in-line comments within them. You may add additional functions as you see fit.
You may leverage the functions in the "Utilities" section if you find it necessary.
You may change N, LEARNING_RATE, BATCH_SIZE, NUM_EPOCHS if it helps you train your 
network. Do not modify the code in the section labeled "Do Not Modify Below."
Code in this section will call your functions, so make sure your implementation
is compatible.

Your code must run (you can test it by clicking "Run" button in the top-left).
The "train" method will train your network over NUM_EPOCHS epochs, and print a 
mean-squared error over the hold-out set after each epoch.

Please feel free to add extra print statements if it helps you debug your code.

This exercise is open-book. You may leverage resources you find on the Internet, 
such as syntax references, mathematical formulae, etc., but you should not adapt 
or otherwise use existing implementation code.
"""

#############################################################
######################### PARAMETERS ########################
#############################################################
N = 1000
LEARNING_RATE = 1
BATCH_SIZE = 5
NUM_EPOCHS = 10
INPUT_WIDTH = 2
HIDDEN_LAYER_WIDTH = 4
OUTPUT_LAYER_WIDTH = 1
INITIAL_HIDDEN_LAYER_WEIGHTS = np.random.random((HIDDEN_LAYER_WIDTH, INPUT_WIDTH))
INITIAL_HIDDEN_LAYER_BIASES = np.random.random((HIDDEN_LAYER_WIDTH, 1))
INITIAL_OUTPUT_LAYER_WEIGHTS = np.random.random(
    (OUTPUT_LAYER_WIDTH, HIDDEN_LAYER_WIDTH)
)
INITIAL_OUTPUT_LAYER_BIASES = np.random.random((OUTPUT_LAYER_WIDTH, 1))

#############################################################
######################### UTILITIES #########################
#############################################################
def sigmoid(z):
    # activation function
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    # derivative of activation function
    return sigmoid(z) * (1 - sigmoid(z))


#############################################################
################## FILL IN THESE FUNCTIONS ##################
#############################################################
def compute_hidden_layer_weighted_input(x, hidden_layer_weights, hidden_layer_biases):
    # return the weighted inputs (before applying sigmoid) for layer 1 as a 4x1 matrix

    # fill in
    # Outputs should be \sum w_{i}*x_{i} + biases
    hidden_layer_outputs = np.dot(x,hidden_layer_weights) + hidden_layer_biases
    return hidden_layer_outputs


def compute_output_layer_weighted_input(
    hidden_layer_activation, output_layer_weights, output_layer_biases
):
    # return the weighted inputs (before applying sigmoid) for output layer as a 1x1 matrix

    # fill in
    # Outputs should be \sum activation(w_{i}*x_{i})*output_weights + biases
    # The outputs from the hidden layer get softmax activation applied to them, then serve as inputs to
    # the output layer, we take the dot product of the the outputs from the activated layer and the
    # weights at the output layer and we add the biases
    output_layer_weighted_sum = np.dot(hidden_layer_activation, output_layer_weights) + output_layer_biases
    return output_layer_weighted_sum


def compute_gradients(
    x,
    y,
    hidden_layer_weights,
    hidden_layer_biases,
    hidden_layer_weighted_input,
    output_layer_weights,
    output_layer_biases,
    output_layer_weighted_input,
):
    # x, y is a single training example
    # for a single training example, return the gradient of loss with respect to each layer's weights and biases
    # return value should be a tuple of lists, where the first element is the list of weight gradients,
    # and the second is the list of bias gradients. the shape of each "gradient" should correspond to the shape of the
    # weight/bias matrix it will be used to update.

    # fill in
    #d/dw = -2*w.(y-(w*x+b)) Compute chanin gradients, Gradient with respect to the weights will bring in an extra term w
    #d/dbiases = 2(y-(w*x+b))    Gradient with respect to the biases is -1
   
    hidden_activations = compute_hidden_layer_weighted_input(x, hidden_layer_weights, hidden_layer_biases)
    outputs = compute_output_layer_weighted_input(sigmoid(hidden_activations), output_layer_weights, output_layer_biases)
    grads_hidden = -2*np.dot(hidden_layer_weights,hidden_activations - hidden_layer_weighted_input)
    grads_out = -2*np.dot(output_layer_weights, outputs - output_layer_weighted_input)

    grads_bias_hidden = -2*(hidden_activations - hidden_layer_weighted_input)
    grads_bias_out = -2*(outputs - output_layer_weighted_input)
    

    weight_gradients = [grads_hidden,grads_out]
    bias_gradients = [grads_bias_hidden,grads_bias_out]

    return weight_gradients, bias_gradients


def get_new_weights_and_biases(
    training_batch,
    hidden_layer_weights,
    hidden_layer_biases,
    output_layer_weights,
    output_layer_biases,
):
    # training_batch is a list of (x, y) training examples
    # return the new weights and biases after processing this batch of data, and according to LEARNING_RATE

    #Use the weights computed from above and do it for the whole dataset
    # fill in
    for i in training_batch:
        hidden_activations = compute_hidden_layer_weighted_input(i[0], hidden_layer_weights, hidden_layer_biases)
        outputs = compute_output_layer_weighted_input(sigmoid(hidden_activations), output_layer_weights, output_layer_biases)
        new_weights, new_biases = compute_gradients(i[0],i[1],hidden_layer_weights,hidden_layer_biases,hidden_activations,
        output_layer_weights, output_layer_biases, outputs)
        #print("Heey",new_weights)
        #print("Heey",new_biases)
    return new_weights, new_biases

#############################################################
#################### DO NOT MODIFY BELOW ####################
#############################################################
def predict(
    x,
    hidden_layer_weights,
    hidden_layer_biases,
    output_layer_weights,
    output_layer_biases,
):
    # make x a column-vector
    x = np.array([x]).T
    hidden_layer_activation = sigmoid(compute_hidden_layer_weighted_input(x, hidden_layer_weights, hidden_layer_biases))
    output_layer_activation = sigmoid(compute_output_layer_weighted_input(hidden_layer_activation, output_layer_weights, output_layer_biases))
    return output_layer_activation[0][0]

def train(
    X,
    Y,
    hidden_layer_weights,
    hidden_layer_biases,
    output_layer_weights,
    output_layer_biases,
):
    # X is an (n by 2) array (n examples, 2 input features)
    # Y is an (n by 1) array (n examples, target class)
    for batch_start in range(0, len(X), BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        x_batch = X[batch_start:batch_end]
        y_batch = Y[batch_start:batch_end]
        batch = list(zip(x_batch, y_batch))

        new_weights, new_biases = get_new_weights_and_biases(
            batch,
            hidden_layer_weights,
            hidden_layer_biases,
            output_layer_weights,
            output_layer_biases,
        )
        hidden_layer_weights, output_layer_weights = new_weights
        hidden_layer_biases, output_layer_biases = new_biases

    # return the final weights and biases
    return (
        [hidden_layer_weights, output_layer_weights],
        [hidden_layer_biases, output_layer_biases],
    )


def compute_mse(
    X_test,
    Y_test,
    hidden_layer_weights,
    hidden_layer_biases,
    output_layer_weights,
    output_layer_biases,
):
    predictions = []
    for x in X_test:
        predictions.append(
            predict(
                x,
                hidden_layer_weights,
                hidden_layer_biases,
                output_layer_weights,
                output_layer_biases,
            )
        )
    y_hat = np.array(predictions)
    return np.mean((y_hat - Y_test) ** 2)


# prepare input data
X = np.random.choice([0, 1], (N, 2))
Y = np.logical_xor(X[:, 0], X[:, 1]) * 1
X = X + 0.1 * np.random.random((N, 2))

# split into train and test
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

# initialize weigths, biases
hidden_layer_weights, hidden_layer_biases = (
    INITIAL_HIDDEN_LAYER_WEIGHTS,
    INITIAL_HIDDEN_LAYER_BIASES,
)
output_layer_weights, output_layer_biases = (
    INITIAL_OUTPUT_LAYER_WEIGHTS,
    INITIAL_OUTPUT_LAYER_BIASES,
)

# train over epochs, calculate MSE at each epoch
for epoch in range(NUM_EPOCHS):
    weights, biases = train(
        X_train,
        Y_train,
        hidden_layer_weights,
        hidden_layer_biases,
        output_layer_weights,
        output_layer_biases,
    )
    hidden_layer_weights, output_layer_weights = weights
    hidden_layer_biases, output_layer_biases = biases
    epoch_mse = compute_mse(
        X_test,
        Y_test,
        hidden_layer_weights,
        hidden_layer_biases,
        output_layer_weights,
        output_layer_biases,
    )
    print(f"MSE (epoch {epoch}):", epoch_mse)

print("done")
