import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import time 
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

    def __init__(self, X, y, hidden_size, num_neurons, X_val=None, y_val=None):
        self.X = X
        self.y = y
        self.X_val = X_val
        self.y_val = y_val
        self.input_size = X.shape[0] #784
        self.output_size = y.shape[0] # 10
        self.hidden_size = hidden_size
        self.num_neurons = num_neurons
        self.weights = self.generate_weights()
        self.biases = self.generate_biases()
        self.z = []
        self.h = []
        # self.init_plot_parameters()


    def generate_weights(self):
        weights = [np.random.rand(self.num_neurons,self.num_neurons) for layers in range(self.hidden_size-1)]
        weight_input = np.random.rand(self.num_neurons,self.input_size) * 0.01
        weights.insert(0,weight_input)
        weight_output = np.random.rand(self.output_size, self.num_neurons) * 0.01
        weights.append(weight_output)
        return weights

    def generate_biases(self):

        # sizes = [785,30,30,30,10]
        # self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # [print(bias.shape) for bias in self.biases]


        biases = [np.random.rand(self.num_neurons,1)*0.01 for layers in range(self.hidden_size)]
        [print(bias.shape) for bias in biases]

        biases_output = np.random.rand(self.output_size,1) * 0.01
        print(biases_output.shape)
        biases.append(biases_output)
        return biases


    def foward(self, h):
        # TODO add bias to everything
        # TODO add a list to keep track each activation function

        # Input will always a fixed image 784 


        # [print(bias.shape) for bias in self.biases]

        self.h.append(h)
        for weight, bias in zip(self.weights[:-1],self.biases[:-1]):
            # print(weight.shape ,bias.shape)
            z = np.dot(weight, h) + bias
            h = self.relu(z)
            self.z.append(z)
            self.h.append(h)
        z = np.dot(self.weights[-1], h) + self.biases[-1]
        yhat = softmax(z)
        # yhat = self.sigmoid(z)
        self.z.append(z)
        self.h.append(yhat)
        return yhat

    def backwards(self, x, y):
        x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        # print(x.shape)
        gradient_w = [0 for w in self.weights]
        gradient_b = [0 for b in self.biases]

        self.foward(x)

        # do backwards prop after doing first a foward prop

        # [print(f"z shape is {z.shape}") for z in self.z]
        # [print(f"h shape is {h.shape}") for h in self.h]

        g = self.grad_MSE(self.h[-1], y) # (yhat - y) derivative of softmax
        w = np.dot(g,self.h[-2].T)
        gradient_w[-1] = w
        gradient_b[-1] = g
        g = np.dot(self.weights[-1].T,g)

        # [print(f"weight shape is {weight.shape}") for weight in self.weights]

        for layer in range(2,self.hidden_size+2):
            # print(f"g shape is {g.shape} and z shape is {self.relu_prime(self.z[-layer]).shape}")
            g =  g * self.relu_prime(self.z[-layer])
            gradient_b[-layer] = g
            gradient_w[-layer] = np.dot(g,self.h[-layer-1].T)
            # print(f"g shape is {g.shape} and weight shape is {self.weights[-layer].T.shape}")
            g = np.dot(self.weights[-layer].T,g) # before we had -layer-1 but now it works as -layer

        return gradient_w, gradient_b



    def SGD(self,batch_size, epochs, epsilon, alpha):
        # randomize training set
        self.epochs = epochs
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
            print(f"Current epoch [{epoch}]")
            for indexes in rounds:
                start, finish = indexes

                # calculate the backprop per one image and then sum all its weights
                # for x, y in zip()

                mini_batch = [shuffled_X[:,start:finish], shuffled_y[:,start:finish]]
                # print(f"Mini batch round {start}, {finish}")
                self.update_mini_batch(mini_batch,epsilon)

                # for size in 100:
                #     gradient_w = self.backwards(shuffled_X[:,start:finish],y)


                # for x in shuffled_X[:,start:finish]:
                #     for y in shuffled_y[:,start:finish]:

                #         print(f"x is {x[start:finish]}")

                #         gradient_w = self.backwards(x,y)
                #         for i,weightVal in enumerate(weight_gradient):
                #             weightVal += gradient_w[i]

                # for i, weight in enumerate(weight_gradient):
                #     self.weights[i] -= epsilon*weight
                # gradient =  grad_CE(weights, shuffled_X[:,start:finish], shuffled_y[:,start:finish], alpha)
                # weights = weights - epsilon * gradient
        # return 0
            # yhat = nn.foward(X_tr)
            # print(size(self.val_data))

            # if self.X_val is not None:
            #     self.plot_learning_curves()
        plt.show()

    def init_plot_parameters(self):
        self.activate = True
        self.CE_tr = []
        self.CE_val = []
        self.PC_tr = []
        self.PC_val = []
        self.fig, self.axs = plt.subplots(2)


    def plot_learning_curves(self):
        plt.axis(xmin=0, xmax=self.epochs)
        yhat_tr = self.foward(self.X)
        yhat_val = self.foward(self.X_val)

        self.CE_tr.append(CE(yhat_tr, self.y))
        self.CE_val.append(CE(yhat_val, self.y_val))
            
        self.PC_tr.append(PC(yhat_tr, self.y))
        self.PC_val.append(PC(yhat_val, self.y_val))


        self.axs[0].plot(self.CE_tr, color='green', linewidth=3, label="train")
        self.axs[0].plot(self.CE_val, color='r', linewidth=2, label="val")
        self.axs[0].set_xlabel("$epoch$", fontsize=12)
        self.axs[0].set_ylabel('{}'.format('Loss'), fontsize=12)

        self.axs[1].plot(self.PC_tr, color='green', linewidth=3, label="train")
        self.axs[1].plot(self.PC_val, color='r', linewidth=2, label="val")
        self.axs[1].set_xlabel("$epoch$", fontsize=12)
        self.axs[1].set_ylabel('{}'.format('PC'), fontsize=12)

        if self.activate is True:
            self.axs[0].legend()
            self.axs[1].legend()
        self.activate =False
        plt.pause(0.05)
        plt.tight_layout()


    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        # print(mini_batch[1].shape)
        for i in range(len(mini_batch)):
            
            delta_nabla_w, delta_nabla_b = self.backwards(mini_batch[0][:,i], mini_batch[1][:,i])
                
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                        for b, nb in zip(self.biases, nabla_b)]


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
        return self.sigmoid(z)*(1-self.sigmoid(z))


# Percent of correctly classified images
def PC(yhat, y):
    return np.mean(np.argmax(yhat, axis = 0) == np.argmax(y, axis=0))

# Calculate Cross Entropy without regularization 
def CE(yhat, y):
    # vectorize formula
    ce = y * np.log(yhat) # (10,5000)
    verticalSum = np.sum(ce, axis=0) # (5000)
    celoss = np.mean(verticalSum) * -1
    return celoss

#INPUT: Z = W * X where W is shape (10,30) and X is shape (30,500)
def softmax(z):
    # z = np.dot(weights, Xtilde) # (10 numOfClasses,100 numOfImages)
    yhat = np.exp(z) / np.sum(np.exp(z), axis=0) # axis=0 means sum rows (10)
    return yhat 

# input matrix has each picture as a column vector (example: 2304 pixels, 5000 examples) 
def append_bias(matrix):
    bias = np.ones(matrix.shape[1]) # 5000
    return np.r_[matrix,[bias]]

def train_number_classifier ():
    # Load data and append bias
    X_tr = np.load("mnist_train_images.npy").T  # (784, 55000)
    y_tr = np.load("mnist_train_labels.npy").T # (10, 55000)
    X_val = np.load("mnist_validation_images.npy").T # (784, 5000)
    y_val = np.load("mnist_validation_labels.npy").T # (10, 5000)   
    X_te = np.load("mnist_test_images.npy").T
    y_te = np.load("mnist_test_labels.npy").T
    
    # Hyper parameters 
    hidden_layers = [3,4,5]
    num_units = [30,40,50] # num of neuros per layer
    mini_batch_sizes = [100, 500, 1000, 2000] # mini batch sizes
    epochs = [1, 2, 3, 4] # number of epochs
    epsilons = [0.1, 3e-3, 1e-3, 3e-5] # learning rates
    alphas = [0.1, 0.01, 0.05, 0.001] # regularization alpha

    nn = NN(X_tr,y_tr, 2, 50, X_val, y_val)
    nn.SGD(16,50,0.0005,0.1)
    yhat = nn.foward(X_tr)
    print(yhat.shape)
    pc_tr = PC(yhat, y_tr)
    print("The PC for training set is " + str(pc_tr) + " correct")


    # def SGD(self,batch_size, epochs, epsilon, alpha):



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

