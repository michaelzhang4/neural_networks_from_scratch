import numpy as np
import helpers as h
import time

class multinomial_logistic_regression:
    # y is array of array of classes with 1 for correct class and 0 if not correct (one hot encoding)
    # e.g. [[1,0,0],[0,1,0],[0,0,1]]
    def __init__(self, X, y, encoding):
        self.y : np.ndarray = y
        self.classes = y.shape[1]
        rng = np.random.default_rng(int(time.time()))
        self.w : np.ndarray = rng.uniform(size=(X[0].size,self.classes))
        self.b : np.ndarray = rng.random(size=(1,self.classes))
        self.X : np.ndarray = X
        self.n : int = len(X)
        self.iterations : int = 50000
        # Learning rate
        self.alpha : float = 0.01
        self.min_values=[]
        self.max_values=[]
        self.encoding = encoding
        self.scale_data()
    
    def train(self):
        for i in range(self.iterations):
            y_hat = h.softmax(np.dot(self.X,self.w)+self.b)
            # Get error matrix (predicted val - real val)
            error = y_hat - self.y
            # Derivatives of weights and bias
            dw = 1/self.n * np.dot(self.X.T,error)
            db = 1/self.n * np.sum(error, axis=0, keepdims=True)
            # Update weights and bias using deriv. * learning rate
            self.w-=self.alpha*dw
            self.b-=self.alpha*db
            if(i%10==0):
                print("Iteration",i,":",self.get_accuracy(y_hat,self.y))
    
    def get_accuracy(self,predicted,y):
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
        for i in range(len(features)):
            features[i]=(features[i]-self.min_values[i])/(self.max_values[i]-self.min_values[i])
        return features
    
    def predict(self,X):
        features = self.scale(np.array(X).astype(float))     
        y_hat = h.softmax(np.dot(features, self.w) + self.b)
        
        return self.encoding[np.argmax(y_hat)]