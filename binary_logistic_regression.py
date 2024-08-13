import numpy as np
import helpers as h
 
class binary_logistic_regression:
    # Weights, bias, alpha, iterations
    def __init__(self,X,y):
        self.w = np.ones(len(X[0]))
        self.b = 1
        self.n = len(X)
        self.X = X
        self.y = y
        self.alpha = 0.01
        self.iterations = 1000

    # Train weights and biases
    def train(self):
        for iter in range(self.iterations):
            # Run on every sample in order in the training dataset
            # Stochastic Gradient Descent
            for i in range(self.n):
                logit=np.dot(self.X[i],self.w.T)+self.b
                y_hat=h.sigmoid(logit)
                if(y_hat==1 or y_hat==0):
                    y_hat+=0.00001
                # Gradient descent on weights
                for j in range(len(self.w)):
                    dw=((1-self.y[i])/(1-y_hat)-(self.y[i]/y_hat))*(y_hat*(1-y_hat))*self.X[i][j]
                    self.w[j]-=self.alpha*dw
                # Same for Bias
                db=((1-self.y[i])/(1-y_hat)-(self.y[i]/y_hat))*(y_hat*(1-y_hat))
                self.b-=self.alpha*db


    def predict(self, features):
        f = np.array(features)
        logit = np.dot(f,self.w.T)+self.b
        y_hat = h.sigmoid(logit)
        print(y_hat)
        if (y_hat >= 0.5):
            return 1
        elif (y_hat < 0.5):
            return 0
