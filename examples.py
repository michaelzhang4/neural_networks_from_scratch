import ordinary_least_regression as OLS
import iterative_least_regression as ILS

def OLS_example():
    X = [5,7,12,16,20]

    y = [40,120,180,210,240]

    OLS.OLS(X,y)
    
    
def IRLS_example():
    X = [5,7,12,16,20]

    y = [40,120,180,210,240]

    ILS.IRLS(X,y)