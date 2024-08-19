
# Numpy for crunching matrix multiplication and matplotlib to view results
import numpy
# Auxiliary helpers for procuring data
import math, random, time, cassiopeia as cass
# For datasets
import sklearn.datasets

import helpers as h


class node:
    def __init__(self,connections, activation):
        self.w = np.ones((connections))
        self.b = 1
        self.activation = activation
        self.init_randomise()

    def init_randomise(self):
        random.seed(time.time())
        for i in range(len(self.w)):
            self.w[i] = random.random()
        self.b = random.random()

    def compute_output(self, X):
        logit = np.dot(self.w.T,X) + self.b
        return (logit,self.apply_activation(logit))

    def apply_activation(self,z):
        if self.activation == 'relu':
            return self.relu(z)
        elif self.activation == 'tanh':
            return self.tanh(z)
        elif self.activation == 'sigmoid':
            return self.sigmoid(z)
        elif self.activation == 'softmax':
            return self.softmax(z)
        else:
            return z
    
    def relu(self, x):
        if x > 0:
            return x
        else:
            return 0

    def sigmoid(self, x):
        return 1/(1+math.e**(-x))

    def tanh(self, x):
        c = math.e**(x)
        nc = math.e**(-x)
        return (c-nc)/(c+nc)

    def softmax(self, x):
        y_hat=[]
        logits=[]
        total=0
        for i in range(len(self.classes)):
            logit = np.dot(self.w[i],x.T)+self.b
            exp = math.e**(logit)
            exps.append(exp)
            total += exp
        for i in range(len(self.classes)):
            y_hat.append(exps[i]/total)
        return y_hat.index(max(y_hat))


# Feed forward neural network
class feed_forward_nn:
    def __init__(self, X, y, encoding,hidden_layers):
        self.X = X
        self.y = y
        self.classes = len(encoding) 
        self.encoding = encoding
        # each item in list is a number of nodes per layer
        # e.g. [7, 9] is 2 layers with 7 and 9 nodes respectively
        self.hidden_layers = hidden_layers
        self.nodes = []
        self.create_layers()
        if self.classes < 2:
            print("Error output classes must be 2 or greater")


    def relu(self, x):
        if x > 0:
            return x
        else:
            return 0

    def sigmoid(self, x):
        return 1/(1+math.e**(-x))

    def tanh(self, x):
        c = math.e**(x)
        nc = math.e**(-x)
        return (c-nc)/(c+nc)

    def softmax(self, x):
        y_hat=[]
        logits=[]
        total=0
        for i in range(self.classes):
            logit = np.dot(self.w[i],x.T)+self.b
            exp = math.e**(logit)
            exps.append(exp)
            total += exp
        for i in range(self.classes):
            y_hat.append(exps[i]/total)
        return y_hat

    def create_layers(self):
        for i,layer_num in enumerate(self.hidden_layers):
            layer = []
            for j in range(layer_num):
                if i == 0:
                    layer.append(node(len(self.X[0]),"None"))
                else:
                    layer.append(node(self.hidden_layers[i-1],"None"))
            self.nodes.append(layer)
        output_layer = []
        if self.classes == 2:
            output_layer.append(node(self.hidden_layers[-1],"sigmoid"))
        else:
            for i in range(self.classes):
                output_layer.append(node(self.hidden_layers[-1],"softmax"))
        self.nodes.append(output_layer)

    def train_sig(self):
        for i in range(len(self.X)):

            outputs = self.forward_pass(self.X[i])
            y_hat = self.sigmoid(outputs[-1])
            errors = []
            output_error = (y_hat-self.y[i])*(self.sigmoid(outputs[-1]))*(1-self.sigmoid(outputs[-1]))
            errors.append(output_error)
            # j is layer number
            for j in range(len(self.nodes)-1):
                # inverse layer number and exclude output layer
                layer = len(self.nodes)-1-j
                for k in range(len(self.nodes[layer])):
                    error = (np.dot(self.nodes[layer][k].w, errors[-1]))*(self.sigmoid(outputs[layer]))*(1-self.sigmoid(outputs[layer]))
                    gradient = error * (outputs[layer-1])
                    self.nodes[j][k].w -= self.alpha * gradient



    def train_soft(self):
        for i in range(len(self.X)):
            outputs = self.forward_pass(self.X[i])
            y_hat = self.softmax(np.array([outputs[-1]]))


    def train(self):
        if self.classes==2:
            self.train_sig()
        else:
            self.train_soft()


    def forward_pass(self, inputs):
        outputs = []
        for layer in range(len(self.nodes)):
            output = []
            for nodes in self.nodes[layer]:
                if layer == 0:
                    # (logit, activation)
                    output.append(nodes.compute_output(inputs))
                else:
                    output.append(nodes.compute_output(np.array(outputs[layer-1])))
            outputs.append(output)
        return outputs


        

class CNN:
    def __init__(self):
        # To do
        self.x=5

def CNN_digits_data():
    digits = sklearn.datasets.load_digits()
    y = digits.target
    X = digits.data

a = feed_forward_nn(np.ones((12,4)),np.ones((12)),["output", "rererere"],[8,6])

