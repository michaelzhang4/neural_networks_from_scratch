import matplotlib.pyplot as plt, numpy as np
import helpers as h

# Ordinary Least Squares 
def OLS(X, y):
    X_mean = h.mean(X)
    y_mean = h.mean(y)
    
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

    h.graph_linear_regression(X, y, y_hat, b, equation)