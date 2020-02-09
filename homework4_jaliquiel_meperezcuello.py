import numpy as np
import math

'''
Questions:
- Should the bias be inside the weights or "+ bias"
'''

np.random.seed(1234)


class Network(object):

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]



# Neural network class
class NN(object):

    def __init__(self, X, y, hidden_size, num_neurons):
        self.X = X
        self.y = y
        self.input_size = X.shape[0] #784
        self.output_size = y.shape[0] # 10
        self.hidden_size = hidden_size
        self.num_neurons = num_neurons
        self.weights = self.generateWeights()

    def generateWeights(self):
        weights = [np.random.rand(self.num_neurons,self.num_neurons) for layers in range(self.hidden_size-1)]
        weight_input = np.random.rand(self.num_neurons,self.input_size) * 0.01
        weights.insert(0,weight_input)
        weight_output = np.random.rand(self.output_size, self.num_neurons) * 0.01
        weights.append(weight_output)
        return weights

    def feed_foward(self):
        # TODO add bias to everything
        # TODO add a list to keep track each activation function
        h = self.X
        for weight in self.weights[:-1]:
            z = np.dot(weight, h)
            h = relu(z)

        yhat = softmax(self.weights[-1], h)
        return yhat


def SGD():
    pass

def relu(z):
    z[ z <= 0] = 0
    return z

#INPUT: Z = W * X where W is shape (10,30) and X is shape (30,500)
def softmax(weights, Xtilde):
    z = np.dot(weights, Xtilde) # (10 numOfClasses,100 numOfImages)
    yhat = np.exp(z) / np.sum(np.exp(z), axis=0) # axis=0 means sum rows (10)
    return yhat 

# input matrix has each picture as a column vector (example: 2304 pixels, 5000 examples) 
def append_bias(matrix):
    bias = np.ones(matrix.shape[1]) # 5000
    return np.r_[matrix,[bias]]

def train_number_classifier ():
    # Load data and append bias
    X_tr = append_bias(np.load("mnist_train_images.npy").T)  # (784, 55000)
    y_tr = np.load("mnist_train_labels.npy").T # (10, 55000)
    X_val = append_bias(np.load("mnist_validation_images.npy").T) # (784, 5000)
    y_val = np.load("mnist_validation_labels.npy").T # (10, 5000)   
    X_te = append_bias(np.load("mnist_test_images.npy").T)
    y_te = np.load("mnist_test_labels.npy").T
    
    # Hyper parameters 
    hidden_layers = [3,4,5]
    num_units = [30,40,50] # num of neuros per layer
    mini_batch_sizes = [100, 500, 1000, 2000] # mini batch sizes
    epochs = [1, 2, 3, 4] # number of epochs
    epsilons = [0.1, 3e-3, 1e-3, 3e-5] # learning rates
    alphas = [0.1, 0.01, 0.05, 0.001] # regularization alpha

    nn = NN(X_tr,y_tr,3, 30)
    yhat= nn.feed_foward()
    print(yhat.shape)



if __name__ == '__main__':
    train_number_classifier()

    # a = Network([785,30,30,30,10])
    # for weight in a.weights:
    #     print(weight.shape)

