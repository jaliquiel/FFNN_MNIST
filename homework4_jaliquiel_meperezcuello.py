import numpy as np
import math

'''
Questions:
- Should the bias be inside the weights or "+ bias" (for gradient)
- Is the gradient of CE does it include 1/n
- Class 6 slide is it doing MSE or is it doing CE? Does the formula stay the same?
- when we go backwards for the first time, is it softmax or relu
- do we need softmax prime??????????
- what do we need to vectorize in th backProp function
'''

np.random.seed(1234)

# return list of tuples (start,end) for slicing each batch X
def get_indexes(n, batchSize):
    indexes = []  # list of (start,end) for slicing each batch X
    index = 0
    for round in range(math.ceil(n / batchSize)):
        index += batchSize
        if index > n:
            index -= batchSize
            indexes.append((index, n))
            break
        indexes.append((index - batchSize, index))
    return indexes


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
        self.z = []
        self.h = []

    def generateWeights(self):
        weights = [np.random.rand(self.num_neurons,self.num_neurons) for layers in range(self.hidden_size-1)]
        weight_input = np.random.rand(self.num_neurons,self.input_size) * 0.01
        weights.insert(0,weight_input)
        weight_output = np.random.rand(self.output_size, self.num_neurons) * 0.01
        weights.append(weight_output)
        return weights

    def foward(self, h):
        # TODO add bias to everything
        # TODO add a list to keep track each activation function

        # Input will always a fixed image 784 
        self.h.append(h)
        for weight in self.weights[:-1]:
            z = np.dot(weight, h)
            h = self.relu(z)
            self.z.append(z)
            self.h.append(h)
        z = np.dot(self.weights[-1], h)
        yhat = softmax(z, h)
        self.z.append(z)
        self.h.append(yhat)
        return yhat

    def backwards(self, x, y):
        gradient_w = [0 for w in self.weights]
        # gradient_b = []

        self.foward(x)

        # do backwards prop after doing first a foward prop

        # [print(f"z shape is {z.shape}") for z in self.z]
        # [print(f"h shape is {h.shape}") for h in self.h]

        g = self.grad_MSE(self.h[-1], y) * self.sigmoidPrime(self.z[-1])
        w = np.dot(g,self.h[-2].T)
        gradient_w[-1] = w

        g = np.dot(self.weights[-1].T,g)

        # [print(f"weight shape is {weight.shape}") for weight in self.weights]

        for layer in range(2,self.hidden_size+2):
            # print(f"g shape is {g.shape} and z shape is {self.relu_prime(self.z[-layer]).shape}")
            g =  g * self.sigmoidPrime(self.z[-layer])
            gradient_w[-layer] = np.dot(g,self.h[-layer-1].T)
            # print(f"g shape is {g.shape} and weight shape is {self.weights[-layer].T.shape}")
            g = np.dot(self.weights[-layer].T,g) # before we had -layer-1 but now it works as -layer

        return gradient_w

    def SGD(self,batch_size, epochs, epsilon, alpha):
        # randomize training set     
        permute = np.random.permutation(self.X.shape[1])
        shuffled_X  = self.X.T[permute].T #(784, 55000)
        shuffled_y = self.y.T[permute].T #(10,55000)

        sample_size = self.X.shape[1] # total batch size

        # get all indexes based on batch size
        rounds = get_indexes(sample_size, batch_size) # list of (start,end) for slicing each batch X

        # # initialize weights to random values with an standard deviation of 0.01
        # weights = np.random.rand(785,10) * 0.01 

        # start iteration loop


        for epoch in range(epochs):
            for indexes in rounds:
                start, finish = indexes

                weight_gradient = [0 for w in self.weights]

                # calculate the backprop per one image and then sum all its weights
                # for x, y in zip()
                for x in shuffled_X[:,start:finish]:
                    for y in shuffled_y[:,start:finish]:

                        print(f"x is {x[start:finish]}")

                        gradient_w = self.backwards(x,y)
                        for i,weightVal in enumerate(weight_gradient):
                            weightVal += gradient_w[i]

                for i, weight in enumerate(weight_gradient):
                    self.weights[i] -= epsilon*weight
                # gradient =  grad_CE(weights, shuffled_X[:,start:finish], shuffled_y[:,start:finish], alpha)
                # weights = weights - epsilon * gradient
        # return 0

    # Calculate the gradient of cross entropy loss
    # TODO REGULARIZATION FOR LATER
    def grad_CE(self, yhat, y):
        distance = yhat - y
        gradient = np.dot(X,distance.T)
        return gradient


    def grad_MSE(self, yhat, y):
        return yhat - y

    def relu(self, z):
        z[ z <= 0] = 0
        return z

    def relu_prime(self, z):
        z[z <= 0] = 0
        z[z > 0] = 1
        return z


    def sigmoid(self, z):
        """The sigmoid function."""
        return 1.0/(1.0+np.exp(-z))

    def sigmoid_prime(self, z):
        """Derivative of the sigmoid function."""
        return sigmoid(z)*(1-sigmoid(z))





# Calculate Cross Entropy without regularization 
def CE(yhat, y):
    # vectorize formula
    ce = y * np.log(yhat) # (10,5000)
    verticalSum = np.sum(ce, axis=0) # (5000)
    celoss = np.mean(verticalSum) * -1
    return celoss

#INPUT: Z = W * X where W is shape (10,30) and X is shape (30,500)
def softmax(z, Xtilde):
    # z = np.dot(weights, Xtilde) # (10 numOfClasses,100 numOfImages)
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

    nn = NN(X_tr,y_tr,1, 30)
    nn.SGD(10,30,3,0.1)


    # yhat= nn.foward(X_tr)
    # yhat=nn.backwards()

    # [print(weight.shape) for weight in nn.weights]
    # [print(f"gradient weight is {weight.shape}") for weight in nn.backwards()]



    # print(yhat.shape)
    # print(f"h shape is {len(nn.h)}")
    # print(f"z shape is {len(nn.z)}")


if __name__ == '__main__':
    train_number_classifier()

    # a = Network([785,30,30,30,10])
    # for weight in a.weights:
    #     print(weight.shape)

