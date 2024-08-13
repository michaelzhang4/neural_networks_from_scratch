import helpers as h

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

    h.graph_linear_regression(X,y,y_hat,m,equation)