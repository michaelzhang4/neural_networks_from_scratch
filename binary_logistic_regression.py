import numpy as np
import helpers as h
 
class binary_logistic_regression:
    # Weights, bias, alpha, iterations
    def __init__(self,X,y):
        self.w : np.ndarray = np.ones(len(X[0]))
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
            epsilon = 1e-10  # A small value to avoid division by zero
            y_hat = np.clip(y_hat, epsilon, 1 - epsilon)

            # Compute the error vector
            error = y_hat-self.y

            # Compute gradient descent
            dw=((1-self.y)/(1-y_hat)-(self.y/y_hat))*(y_hat*(1-y_hat))*self.X
            db=np.sum(error) / self.n

            # Change weights and bias valyes
            self.w-=self.alpha*dw
            self.b-=self.alpha*db


    def predict(self, features):
        f = np.array(features)
        logit = np.dot(f,self.w)+self.b
        y_hat = h.sigmoid(logit)
        print(y_hat)
        if (y_hat[0] >= 0.5):
            return 1
        elif (y_hat[0] < 0.5):
            return 0
