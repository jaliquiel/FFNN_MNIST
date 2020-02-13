import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation
import time 
import math

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
        c = (self.num_neurons ** (-0.5))/2
        c1 = (self.output_size ** (-0.5))/2

        weights = [np.random.uniform(-c,c,size=(self.num_neurons,self.num_neurons)) for layers in range(self.hidden_size-1)]
        weight_input = np.random.uniform(-c,c,size=(self.num_neurons,self.input_size))
        weights.insert(0,weight_input)
        weight_output = np.random.uniform(-c1,c1,size=(self.output_size, self.num_neurons)) 
        weights.append(weight_output)

        # weights = [np.random.rand(self.num_neurons,self.num_neurons) for layers in range(self.hidden_size-1)]
        # weight_input = np.random.rand(self.num_neurons,self.input_size) * 0.01
        # weights.insert(0,weight_input)
        # weight_output = np.random.rand(self.output_size, self.num_neurons) * 0.01
        # weights.append(weight_output)
        return weights

    def generate_biases(self):
        biases = [np.random.rand(self.num_neurons).reshape(-1,1)*0.01 for layers in range(self.hidden_size)]
        biases_output = np.random.rand(self.output_size).reshape(-1,1) * 0.01
        biases.append(biases_output)
        return biases

    def foward(self, h):
        self.h = []
        self.z = []
        self.h.append(h)
        for weight, bias in zip(self.weights[:-1],self.biases[:-1]):
            z = np.dot(weight, h) + bias
            h = self.relu(z)
            self.z.append(z)
            self.h.append(h)
        z = np.dot(self.weights[-1], h) + self.biases[-1]
        yhat = softmax(z)
        self.z.append(z)
        self.h.append(yhat)
        return yhat

    def backwards(self, X, y, alpha):
        gradient_w = [0 for w in self.weights]
        gradient_b = [0 for b in self.biases]

        yhat = self.foward(X)

        # Use algorithm 6.4 from slides
        g = yhat - y
        w = np.dot(g,self.h[-2].T) 
        gradient_w[-1] = w + (alpha * w) 
        gradient_b[-1] = g
        g = np.dot(self.weights[-1].T,g)

        for layer in range(2,self.hidden_size+2):
            g =  g * self.relu_prime(self.z[-layer])
            gradient_b[-layer] = g
            gradient_w[-layer] = np.dot(g,self.h[-layer-1].T) + (alpha * self.weights[-layer]) 
            g = np.dot(self.weights[-layer].T,g) 

        gradient_b = [g.mean(axis=1).reshape(-1,1) for g in gradient_b]
        
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

        # start iteration loop
        for epoch in range(epochs):
            # print(f"Epoch [{epoch}]")
            for indexes in rounds:
                start, finish = indexes
                delta_nabla_w, delta_nabla_b = self.backwards(shuffled_X[:,start:finish], shuffled_y[:,start:finish], alpha)
                self.weights = [w - epsilon*nw for w, nw in zip(self.weights, delta_nabla_w)]
                self.biases = [b - epsilon*nb for b, nb in zip(self.biases, delta_nabla_b)]

            # todo  UNCOMMENT THIS TO PRINT TFHE LAST 20 EPOCHS
            # if epoch >= (epochs - 20):
            #     yhat_tr = self.foward(self.X)
            #     CE_tr = CE(yhat_tr, self.y)
            #     PC_tr = PC(yhat_tr, self.y)
            #     print("This is epoch round [{}]".format(epoch))
            #     print("Epoch [{}], Cross Entropy Loss: {}".format(epoch , CE_tr))
            #     print("Epoch [{}], Percent Correct: {}".format(epoch, PC_tr))

            # if self.X_val is not None:
            #     self.plot_learning_curves(epoch)
        # plt.show()

    def init_plot_parameters(self):
        self.activate = True
        self.CE_tr = []
        self.CE_val = []
        self.PC_tr = []
        self.PC_val = []
        self.fig, self.axs = plt.subplots(2)

    def plot_learning_curves(self, epoch):
        # plt.axis(xmin=0, xmax=self.epochs)
        yhat_tr = self.foward(self.X)
        yhat_val = self.foward(self.X_val)

        # CE_tr = CE(yhat_tr, self.y)
        # CE_val = CE(yhat_val, self.y_val)
        
        PC_tr = PC(yhat_tr, self.y)
        PC_val = PC(yhat_val, self.y_val)
        print(f"My training data acc is [{PC_tr}], my validation acc is [{PC_val}]")

        # self.axs[0].plot(epoch, CE_tr, marker= 'o', color='green', linewidth=3, label="train")
        # self.axs[0].plot(epoch, CE_val,  marker= 'o', color='r', linewidth=2, label="val")
        # self.axs[0].set_xlabel("$epoch$", fontsize=12)
        # self.axs[0].set_ylabel('{}'.format('Loss'), fontsize=12)

        # self.axs[1].plot(epoch, PC_tr,  marker= 'o', color='green', linewidth=3, label="train")
        # self.axs[1].plot(epoch, PC_val,  marker= 'o', color='r', linewidth=2, label="val")
        # self.axs[1].set_xlabel("$epoch$", fontsize=12)
        # self.axs[1].set_ylabel('{}'.format('PC'), fontsize=12)

        # if self.activate is True:
        #     self.axs[0].legend()
        #     self.axs[1].legend()
        # self.activate =False
        # plt.pause(0.05)
        # plt.tight_layout()

    def plot_learning_curves_save_MEM(self):
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
        
    # Calculate the gradient of cross entropy loss
    def grad_CE(self, x, yhat, y):
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

#INPUT: Z = W * X where W is shape (numberOfClasses,numberOfNeurons) and X is shape (numberOfneurons,numberOfimages)
# (10,30) (30,500) Z = ((10 numOfClasses,500 numOfImages))
def softmax(z):
    yhat = np.exp(z) / np.sum(np.exp(z), axis=0) # axis=0 means sum rows (10)
    return yhat 

# Find the best hyper parameters and train multiple Neural networks
def findBestHyperparameters(X_tr, y_tr, X_val, y_val):
    # Hyper parameters 
    hidden_layers = [1,3]
    num_units = [30,40] # num of neuros per layer
    mini_batch_sizes = [100,10] # mini batch sizes
    epochs = [15,30] # number of epochs
    epsilons = [0.001, 0.01] # learning rates, 0.001
    alphas = [0.1,0.01] # regularization alpha

    # hidden_layers = [3]
    # num_units = [40] # num of neuros per layer
    # mini_batch_sizes = [100] # mini batch sizes
    # epochs = [30] # number of epochs
    # epsilons = [0.001] # learning rates, 0.001
    # alphas = [0.1] # regularization alpha

    # Lists to store our all the different hyperparameter sets, their weights and their MSE
    hyper_param_grid = []
    neural_network_param = []
    count = 0
    best_accuracy = 0

    # train weights based on all the different sets of hyperparameters
    for hidden_layer in hidden_layers:
        for num_unit in num_units:
            for mini_batch_size in mini_batch_sizes:
                for epoch in epochs:
                    for epsilon in epsilons:
                        for alpha in alphas:

                            neural_network = NN(X_tr,y_tr,hidden_layer,num_unit, X_val,y_val)
                            neural_network.SGD(mini_batch_size,epoch,epsilon, alpha)

                            # calculate the CE and PC with the validation set
                            yhat = neural_network.foward(X_val)
                            ceVal = CE(yhat, y_val)
                            pcVal = PC(yhat, y_val)

                            if pcVal > best_accuracy:
                                best_accuracy = pcVal
                                neural_network_param = [hidden_layer, num_unit, np.copy(neural_network.weights), np.copy(neural_network.biases)]
                                hyper_param_grid = [mini_batch_size, epoch, epsilon, alpha, pcVal]

                            count += 1
                            print("The CE for [" + str(count) + "] validation set is " + str(ceVal))
                            print("The PC for [" + str(count) + "] validation set is " + str(pcVal) + " correct")
                            print("hidden layers: {}, number of neurons: {}, miniBatch: {}, epoch: {}, epsilon: {}, alpha: {}".format(hidden_layer ,num_unit,mini_batch_size,epoch,epsilon,alpha))
    return hyper_param_grid, neural_network_param

def train_number_classifier ():
    # Load data and append bias
    X_tr = np.load("mnist_train_images.npy").T  # (784, 55000)
    y_tr = np.load("mnist_train_labels.npy").T # (10, 55000)
    X_val = np.load("mnist_validation_images.npy").T # (784, 5000)
    y_val = np.load("mnist_validation_labels.npy").T # (10, 5000)   
    X_te = np.load("mnist_test_images.npy").T
    y_te = np.load("mnist_test_labels.npy").T
    
    # TESTED HYPERPARAMETERS
    # The PC for [2] validation set is 0.9426 correct
    # hidden layers: 2, number of neurons: 30, miniBatch: 16, epoch: 50, epsilon: 0.001, alpha: 0.01

    # The PC for [6] validation set is 0.938 correct
    # hidden layers: 2, number of neurons: 30, miniBatch: 16, epoch: 100, epsilon: 0.001, alpha: 0.01

    # The PC for [14] validation set is 0.9266 correct
    # hidden layers: 2, number of neurons: 30, miniBatch: 32, epoch: 100, epsilon: 0.001, alpha: 0.01 

    # The PC for [16] validation set is 0.907 correct
    # hidden layers: 2, number of neurons: 30, miniBatch: 32, epoch: 100, epsilon: 0.01, alpha: 0.01

    # My best hyperparameters were: (old weight initialization)
    # Hidden Layer Size: 1, Number of Neurons: 50, Mini Batch Size: 100, epoch: 15, epsilon: 0.001, alpha: 0
    # The PC for test set is 0.9671% correct

    # My best hyperparameters were:
    # Hidden Layer Size: 2, Number of Neurons: 50, Mini Batch Size: 100, epoch: 15, epsilon: 0.0001, alpha: 0
    # The PC for test set is 0.9116% correct

    # new weights ====================================== 
    # My best hyperparameters were: 
    # Hidden Layer Size: 3, Number of Neurons: 30, Mini Batch Size: 10, epoch: 30, epsilon: 0.0001, alpha: 0
    # The PC for test set is 0.9355% correct

    # My best hyperparameters were:
    # Hidden Layer Size: 1, Number of Neurons: 40, Mini Batch Size: 10, epoch: 15, epsilon: 0.01, alpha: 0
    # The PC for test set is 0.9697% correct

    # My best hyperparameters were:
    # Hidden Layer Size: 3, Number of Neurons: 40, Mini Batch Size: 100, epoch: 30, epsilon: 0.001, alpha: 0.1
    # The CE for test set is 0.10139482034664093
    # The PC for test set is 0.9715 correct

    hyper_param_grid, neural_network_param = findBestHyperparameters(X_tr, y_tr, X_val, y_val)

    # show the best hyperparameters
    print("--------------------------------------------------------")
    print("My best hyperparameters were: ")
    print("Hidden Layer Size: {}, Number of Neurons: {}, Mini Batch Size: {}, epoch: {}, epsilon: {}, alpha: {}".format(neural_network_param[0], neural_network_param[1],hyper_param_grid[0], hyper_param_grid[1], hyper_param_grid[2], hyper_param_grid[3]))

    # create neural network with our best hyperparameters, weights and biases (no need to train)
    best_neural_network = NN(X_tr,y_tr, neural_network_param[0], neural_network_param[1], X_val, y_val)
    best_neural_network.weights = neural_network_param[2] # weights
    best_neural_network.biases = neural_network_param[3] # biases

    # # Report CE cost on the training
    best_yhat = best_neural_network.foward(X_te)
    ce_te = CE(best_yhat, y_te)
    pc_te = PC(best_yhat, y_te)
    print("The CE for test set is " + str(ce_te))
    print("The PC for test set is " + str(pc_te) + " correct")


if __name__ == '__main__':
    train_number_classifier()

