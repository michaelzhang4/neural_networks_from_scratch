import numpy as np
import helpers as h
import time

class feed_forward_nn:
    # input layer > hidden layer with Relu activation > output layer with Softmax activation 
    def __init__(self, X, y, encoding):
        self.y : np.ndarray = y
        self.classes = y.shape[1]
        rng = np.random.default_rng(int(time.time()))
        self.w1 : np.ndarray = rng.uniform(size=(X[0].size,self.classes))
        self.w2 : np.ndarray = rng.uniform(size=(self.classes,self.classes))
        self.b1 : np.ndarray = rng.random(size=(1,self.classes))
        self.b2 : np.ndarray = rng.random(size=(1,self.classes))
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
            logit1 = np.dot(self.X,self.w1)+self.b1
            ans1 = h.relu(logit1)
            logit2 = np.dot(ans1,self.w2)+self.b2
            y_hat = h.softmax(logit2)
            # Get error matrix (predicted val - real val)
            error = y_hat - self.y
            # Maybe not .T
            dw2 = 1/self.n * np.dot(ans1.T,error)
            db2 = 1/self.n * np.sum(error, axis=0, keepdims=True)
            relu_error = np.dot(error,self.w2.T) * (logit1>0)
            dw1 = 1/self.n * np.dot(self.X.T,relu_error)
            db1 = 1/self.n * np.sum(relu_error, axis=0, keepdims=True)
            self.w1-=self.alpha*dw1
            self.w2-=self.alpha*dw2
            self.b1-=self.alpha*db1
            self.b2-=self.alpha*db2
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
        y_hat = h.softmax(np.dot(h.relu(np.dot(features, self.w1) + self.b1),self.w2)+self.b2)
        
        return self.encoding[np.argmax(y_hat)]