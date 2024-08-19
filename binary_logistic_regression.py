import numpy as np
import helpers as h
import time
 
class binary_logistic_regression:
    # Weights, bias, alpha, iterations
    def __init__(self,X,y):
        rng = np.random.default_rng(int(time.time()))
        self.w : np.ndarray = rng.uniform(size=X[0].size)
        self.b : float = 1
        self.X : np.ndarray = X
        self.y : np.ndarray = y
        self.n : int = len(X)
        self.alpha : float = 0.01
        self.iterations : int = 1000

    # Train weights and biases
    def train(self):
        for iter in range(self.iterations):
            # Run on every sample in order in the training dataset
            # Stochastic Gradient Descent
            logits=np.dot(self.X,self.w)+self.b
            y_hat=h.sigmoid(logits)
            # epsilon = 1e-10  # A small value to avoid division by zero
            # y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

            # Compute the error vector
            error = y_hat-self.y

            # Compute gradient descent
            dw=np.sum(self.X*error.reshape(error.size,1),axis=0)
            db=np.sum(error)/self.n

            # Change weights and bias values
            self.w-=self.alpha*dw
            self.b-=self.alpha*db


    def predict(self, features):
        f = np.array(features)
        logit = np.dot(f,self.w)+self.b
        y_hat = h.sigmoid(logit)
        if (y_hat >= 0.5):
            return "true"
        elif (y_hat < 0.5):
            return "false"
