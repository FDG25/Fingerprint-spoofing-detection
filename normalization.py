import main
import numpy

def centeringData(DTR):
    mu = DTR.mean(1)
    return DTR - main.vcol(mu)

def standardizingData(DTR):
    standard_deviation = numpy.sqrt(numpy.var(DTR, axis=1))
    S_DTR = numpy.zeros(DTR.shape) 
    for i in range(0,DTR.shape[1]):
        S_DTR[:,i] = DTR[:,i]/standard_deviation
    return S_DTR

def zNormalizingData(DTR):
    C_DTR = centeringData(DTR)
    return standardizingData(C_DTR)

def whiteningData(DTR):
    _,cov = main.computeMeanCovMatrix(DTR)
    # Step 1: Compute SVD of the covariance matrix
    U, s, V = numpy.linalg.svd(cov)
    # Step 2: Calculate inverse square root of singular values
    inv_sqrt_s = numpy.diag(1.0 / numpy.sqrt(s))
    # Step 3: Multiply inverse square root of singular values with transposed U matrix
    whitening_matrix = numpy.dot(inv_sqrt_s, U.T)
    # Step 4: Multiply with V matrix to obtain whitening transformation matrix
    whitening_matrix = numpy.dot(whitening_matrix, V)
    # Step 5: Whiten the covariance matrix
    whitened_cov = numpy.dot(whitening_matrix, numpy.dot(cov, whitening_matrix.T))
    # Apply Whitened CovMatrix to every sample
    W_DTR = numpy.zeros(DTR.shape) 
    for i in range(0,DTR.shape[1]):
        W_DTR[:,i] = numpy.dot(DTR[:,i],whitened_cov)
    return W_DTR

def l2NormalizingData(DTR):
    L2_DTR = numpy.zeros(DTR.shape) 
    for i in range(0,DTR.shape[1]):
       L2_DTR[:,i] = DTR[:,i]/numpy.linalg.norm(DTR[:,i])
    return L2_DTR