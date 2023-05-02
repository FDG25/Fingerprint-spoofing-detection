import numpy
import main

def PCA_projection(D,m):
    
    _, C = main.computeMeanCovMatrix(D)

    s, U = numpy.linalg.eigh(C)

    P = U[:, ::-1][:, 0:m]
    
    DP = numpy.dot(P.T, D)
    
    # return the projected dataset
    return DP