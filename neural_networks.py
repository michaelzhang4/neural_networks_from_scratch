import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math

plt.style.use('ggplot')

# Calculates the mean of a variable
def mean(var):
    mean=0
    for elem in var:
        mean+=elem
    mean/=len(var)
    return mean

# Mean Squared Error
def MSE(m,c,X,y):
    # Residual loss
    residual=0
    n = len(X)
    for i in range(n):
        residual+=(y[i]-(X[i]*m+c))**2
    return residual/n

# Plots a visualization of simple linear regression
def OLS_graph(X, y, y_hat, slope, equation):
    
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

# Ordinary Least Squares 
def OLS(X, y):
    X_mean = mean(X)
    y_mean = mean(y)
    
    numerator = 0;
    denominator = 0;

    for i in range(len(X)):
        numerator+=(X[i]-X_mean)*(y[i]-y_mean)

    for i in range(len(X)):
        denominator+=(X[i]-X_mean)**2

    # Regression line is y = bx + a
    b = numerator/denominator
    a = y_mean - b*X_mean

    equation = "y^ = "+str(round(b,3))+"x + "+str(round(a,3))

    y_hat = []

    for elem in X:
        y_hat.append(a + b*elem)

    OLS_graph(X, y, y_hat, b, equation)


def OLS_example():
    X = [5,7,12,16,20]

    y = [40,120,180,210,240]

    OLS(X,y)


# Iteratively Reweighted Least Squares
def IRLS(X, y):
    # Initial gradient and y-intercept of line
    m=1
    c=0
    n=len(X)
    alpha=0.001
    
    iterations = 10000

    for _ in range(iterations):
        dm = 0
        dc = 0
        for i in range(n):
            dm+=(y[i]-(m*X[i]+c))*(X[i])
            dc+=(y[i]-(m*X[i]+c))
        dm=dm*(-2/n)
        dc=dc*(-2/n)
        m=m-dm*alpha
        c=c-dc*alpha
    y_hat=[]
    for i in range(n):
        y_hat.append(X[i]*m+c)
    equation = "y^ = "+str(round(m,3))+"x + "+str(round(c,3))

    OLS_graph(X,y,y_hat,m,equation)
    
def IRLS_example():
    X = [5,7,12,16,20]

    y = [40,120,180,210,240]

    IRLS(X,y)

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

    # Sigmoid function
    def sigmoid(self,x):
        return 1/(1+math.e**(-x))

    # Log loss function
    def log_loss(self,y,y_hat):
        e=0.0001
        return -(y*math.log(y_hat+e)+(1-y)*math.log(1-y_hat+e))

    # Train weights and biases
    def train(self):
        for iter in range(self.iterations):
            # Run on every sample in order in the training dataset
            # Stochastic Gradient Descent
            for i in range(self.n):
                logit=np.dot(self.X[i],self.w.T)+self.b
                y_hat=self.sigmoid(logit)
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
        y_hat = self.sigmoid(logit)
        print(y_hat)
        if (y_hat >= 0.5):
            return 1
        elif (y_hat < 0.5):
            return 0

def binary_data():
    X=np.array([[1,1,1,18],[0,0,0,20],[0,0,0,28],[0,0,0,25],[0,1,1,23],
                [1,1,0,20],[1,1,1,20],[0,0,0,20],[0,1,0,20],[0,1,1,23],
                [1,1,1,23],[1,1,0,23],[0,0,1,20],[1,1,1,1],[1,1,1,13],
                [1,0,1,14],[1,0,0,16],[0,1,1,17],[0,1,1,18],[1,0,1,18],
                [0,0,1,18],[1,1,1,7],[1,1,1,40],[1,1,1,30],[1,1,1,35],
                [1,0,1,32],[0,0,1,40],[0,0,1,60]])
    y=[1,0,0,0,1,
       0,1,0,0,1,
       1,0,1,0,0,
       0,0,0,1,1,
       1,0,0,1,1,
       1,0,0]
    return (X,y)
    
class multinomial_logistic_regression:
    def __init__(self, X, y):
        self.y = y
        self.classes = len(y)
        self.w = np.ones((self.classes,len(X[0])))
        self.b = 1
        self.X = X
        self.n = len(X)
        self.iterations = 1000
        self.alpha = 0.01
    
    def softmax(self,X):
        values = []
        total=0
        y_hat=[]
        for i in range(self.classes):
            weights = self.w[i]
            logit = np.dot(weights.T,X)+self.b
            value = math.e**(logit)
            values.append(value)
            total=value
        for i in range(self.classes):
            y_hat.append(values[i]/total)
        return y_hat

X=np.array([[1,2,3],[2,3,4]])
y=np.array([4,9])


mlr = multinomial_logistic_regression(X,y)

print(mlr.softmax(X[0]))
