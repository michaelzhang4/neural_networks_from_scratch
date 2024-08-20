import numpy as np
import helpers as h
import time

class feed_forward_nn:
    # input layer > hidden layer with Relu activation > output layer with Softmax activation 
    def __init__(self, X, y, encoding):
        # Generate random seed
        rng = np.random.default_rng(int(time.time()))
        # One hot encoded answer array
        self.y : np.ndarray = y#self.one_hot_encode(y,np.max(y))
        # Number of classes
        self.classes = y.shape[1]
        # Weights 1 - before relu, initialised with a uniform distribution
        self.w1 = rng.uniform(-1, 1, size=(X[0].size, self.classes)) / np.sqrt(X[0].size)
        # Weights 2 - after relu, initialised with a uniform distribution
        self.w2 = rng.uniform(-1, 1, size=(self.classes, self.classes)) / np.sqrt(self.classes)
        # Bias 1 - ""
        self.b1 : np.ndarray = np.zeros((1,self.classes))
        # Bias 2 - ""
        self.b2 : np.ndarray = np.zeros((1,self.classes))
        # Input data array
        self.X : np.ndarray = X
        # Length of input data
        self.n : int = len(X)
        # Number of iterations for training
        self.iterations : int = 10000
        # Learning rate
        self.alpha : float = 0.005
        # Encoding array for labelling outputs
        self.encoding = encoding

    # Train the weights and biases of the model using forward pass + backward propogation    
    def train(self):
        for i in range(self.iterations):
            # Forward Pass
            # First logit calculated before relu
            logit1 = np.dot(self.X,self.w1)+self.b1
            # Pass first logit through relu activation function
            ans1 = h.relu(logit1)
            # Use outputs from relu as new inputs which will be class size x class size, calculate second logit
            logit2 = np.dot(ans1,self.w2)+self.b2
            # Use softmax to score the probabilities of each class using the second logit
            y_hat = h.softmax(logit2)

            # Backward propogation
            # Get error matrix (predicted val - real val)
            error = y_hat - self.y
            # Derivative of second weights using error and calculated answers from relu
            dw2 = 1/self.n * np.dot(ans1.T,error)
            # Derivative of second bias using average of errors for each class
            db2 = 1/self.n * np.sum(error, axis=0, keepdims=True)
            # Relu error - dot product of second weights and error matrix * derivative of relu function
            relu_error = np.dot(error,self.w2.T) * (logit1>0)
            # Derivative of first weights using relu_error and input array
            dw1 = 1/self.n * np.dot(self.X.T,relu_error)
            # Derivative of first bias using average relu_error for each class
            db1 = 1/self.n * np.sum(relu_error, axis=0, keepdims=True)

            # Update weights and biases using learning rate
            self.w1-=self.alpha*dw1
            self.w2-=self.alpha*dw2
            self.b1-=self.alpha*db1
            self.b2-=self.alpha*db2

            # Print accuracy information while training
            if(i%100==0):
                print("Iteration",i,":",self.get_accuracy(y_hat,self.y))
    
    def get_accuracy(self,predicted,y):
        # Returns percentage of correct label predictions
        predicted_labels = np.argmax(predicted,axis=1)
        true_labels = np.argmax(y,axis=1)
        return np.sum(true_labels==predicted_labels)/y.shape[0]
   
    def scale_data(self):
        # Finds the minimum and maximum values, then uses the scale all data to between 0 and 1
        self.min_values = np.min(self.X, axis=0)
        self.max_values = np.max(self.X, axis=0)
        
        # Add epsilon to avoid division by zero
        epsilon = 1e-10
        self.X = (self.X - self.min_values) / (self.max_values - self.min_values + epsilon)
    
    def one_hot_encode(self, labels, num_classes):
        return np.eye(num_classes)[labels]


    def scale(self,features):
        # Scales features between 0 and 1
        for i in range(len(features)):
            features[i]=(features[i]-self.min_values[i])/(self.max_values[i]-self.min_values[i])
        return features
    
    def predict(self,X):
        # Returns the highest probability class
        # features = self.scale(np.array(X).astype(float))     
        y_hat = h.softmax(np.dot(h.relu(np.dot(X, self.w1) + self.b1),self.w2)+self.b2)
        
        return self.encoding[np.argmax(y_hat)]