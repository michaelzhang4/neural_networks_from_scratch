import numpy as np
import helpers as h
import time

class feed_forward_nn:
    # input layer > hidden layer with Relu activation > output layer with Softmax activation 
    def __init__(self, X, y, encoding):
        # Generate random seed
        rng = np.random.default_rng(int(time.time()))
        # One hot encoded answer array
        self.y : np.ndarray = y
        # Number of classes
        self.classes = y.shape[1]
        # Weights 1 - before relu
        self.w1 : np.ndarray = rng.uniform(size=(X[0].size,self.classes))
        # Weights 2 - after relu
        self.w2 : np.ndarray = rng.uniform(size=(self.classes,self.classes))
        # Bias 1 - ""
        self.b1 : np.ndarray = rng.random(size=(1,self.classes))
        # Bias 2 - ""
        self.b2 : np.ndarray = rng.random(size=(1,self.classes))
        # Input data array
        self.X : np.ndarray = X
        # Length of input data
        self.n : int = len(X)
        # Number of iterations need ~20000 I found
        self.iterations : int = 50000
        # Learning rate
        self.alpha : float = 0.01
        # Encoding array for labelling outputs
        self.encoding = encoding
        # Structures/function used for scaling the data between 0 and 1
        self.min_values=[]
        self.max_values=[]
        self.scale_data()

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
        max_values=[]
        min_values=[]
        self.X=self.X.astype(float)
        for col in range(len(self.X[0])):
            max_val = 0
            min_val = 99999
            for row in range(len(self.X)):
                if self.X[row][col] > max_val:
                    max_val = self.X[row][col]
                if self.X[row][col] < min_val:
                    min_val = self.X[row][col]
            max_values.append(max_val)
            min_values.append(min_val)
        scale_factors=[]
        for i in range(len(max_values)):
            count=0
            m = c = max_values[i]-min_values[i]
            while(c>10):
                c%=10
                count+=1
            scale_factors.append(round(m+10**count,-count))
        for col in range(len(self.X[0])):
            for row in range(len(self.X)):
                self.X[row][col]= (self.X[row][col]-min_values[col])/(max_values[col]-min_values[col])

        self.max_values=max_values
        self.min_values=min_values

    def scale(self,features):
        # Scales features between 0 and 1
        for i in range(len(features)):
            features[i]=(features[i]-self.min_values[i])/(self.max_values[i]-self.min_values[i])
        return features
    
    def predict(self,X):
        # Returns the highest probability class
        features = self.scale(np.array(X).astype(float))     
        y_hat = h.softmax(np.dot(h.relu(np.dot(features, self.w1) + self.b1),self.w2)+self.b2)
        
        return self.encoding[np.argmax(y_hat)]