import matplotlib.pyplot as plt, numpy as np
import helpers as h

# Ordinary Least Squares 
def OLS(X, y):
    # Calculate the mean of the independent variable X and dependent variable y
    X_mean = h.mean(X)
    y_mean = h.mean(y)
    
    # Initialize variables for the numerator and denominator in the slope calculation
    numerator = 0;
    denominator = 0;

    # Calculate the numerator of the slope (b)
    # This represents the covariance between X and y
    for i in range(len(X)):
        numerator+=(X[i]-X_mean)*(y[i]-y_mean)

    # Calculate the denominator of the slope (b)
    # This represents the variance of X
    for i in range(len(X)):
        denominator+=(X[i]-X_mean)**2

     # Calculate the slope (b) and intercept (a) of the regression line y = bx + a
    b = numerator/denominator # Gradient
    a = y_mean - b*X_mean # Intercept

    # Create a string representation of the regression equation for display
    equation = "y^ = "+str(round(b,3))+"x + "+str(round(a,3))

     # Calculate the predicted y values (ŷ) using the regression line equation
    y_hat = []
    for elem in X:
        y_hat.append(a + b*elem)

    # Plot the original data points and the regression line
    h.graph_linear_regression(X, y, y_hat, b, equation)