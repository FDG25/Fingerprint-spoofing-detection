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
    pass

def l2NormalizingData(DTR):
    L2_DTR = numpy.zeros(DTR.shape) 
    for i in range(0,DTR.shape[1]):
       L2_DTR[:,i] = DTR[:,i]/numpy.linalg.norm(DTR[:,i])
    return L2_DTR