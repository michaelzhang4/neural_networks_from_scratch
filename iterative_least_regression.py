import helpers as h

# Iteratively Reweighted Least Squares
def IRLS(X, y):
    # Initial gradient and y-intercept of line
    m=1
    c=0
    n=len(X)
    # Learning rate
    alpha=0.001
    iterations = 10000

    for _ in range(iterations):
        dm = 0
        dc = 0
        # Use formula to calculate the derivatives of m (gradient) and c (intercept)
        for i in range(n):
            dm+=(y[i]-(m*X[i]+c))*(X[i])
            dc+=(y[i]-(m*X[i]+c))
        dm=dm*(-2/n)
        dc=dc*(-2/n)
        
        # Iterate m and c values
        m=m-dm*alpha
        c=c-dc*alpha
    
    # y predictions
    y_hat=[]
    for i in range(n):
        y_hat.append(X[i]*m+c)
        
    # y = mx + c
    equation = "y^ = "+str(round(m,3))+"x + "+str(round(c,3))

    # Graph equation
    h.graph_linear_regression(X,y,y_hat,m,equation)