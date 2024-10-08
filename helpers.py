import math, numpy as np
import matplotlib.pyplot as plt

# Calculates the mean of a list
def mean(_list):
    mean=0
    for elem in _list:
        mean+=elem
    mean/=len(_list)
    return mean

# Mean Squared Error
def MSE(m,c,X,y):
    # Residual loss
    # y is the true prediction values
    # X contains our data
    # m is gradient, c is intecept which are used to calculated our predicted y
    # then we find the loss average between our predictions and the true values
    residual=0
    n = len(X)
    for i in range(n):
        residual+=(y[i]-(X[i]*m+c))**2
    return residual/n

# Plots a visualization of simple linear regression
def graph_linear_regression(X, y, y_hat, slope, equation):
    # X is 
    fig, ax = plt.subplots()
    ax.axline((X[0],y_hat[0]),slope=slope)
    ax.scatter(X, y)
    
    ax.set_xlim(left=min(0, min(X)), right=round(max(X)*1.2,0))
    ax.set_ylim(bottom=min(0, min(y)), top=round(max(y)*1.1,-2))
    ax.annotate(equation,xy=(X[-2],y_hat[-2]),xytext=(X[-2]*0.7,y_hat[-2]*1.1)) 
    ax.set_title("Least Squares Linear Regression")
    ax.set_ylabel("dependent variable y")
    ax.set_xlabel("independent variable x")
    plt.show()
    
# Sigmoid function
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1/(1+np.exp(-z))

# Softmax function
def softmax(logits: np.ndarray) -> np.ndarray:
    exponentials = np.exp(logits)
    if(logits[0].size==1):
        # Sum row if only one row
        total = np.sum(exponentials)
    else:
        # Sum all rows
        total = np.sum(exponentials,axis=1)
    # This calculates y_hat, the predicted probabilities for each logit e.g. 3 classes [0.4, 0.32, 0.28], which sum to 1
    return exponentials/total.reshape(total.size,1)

# Relu function
def relu(logits: np.ndarray) -> np.ndarray:
    return np.maximum(logits,0)

# Log loss function
def log_loss(self,y,y_hat):
    e=0.0001
    return -(y*math.log(y_hat+e)+(1-y)*math.log(1-y_hat+e))


