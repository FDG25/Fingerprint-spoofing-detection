import numpy
import scipy
import scipy.linalg
from scipy import special
import main
import constants
import lda

def computeNumCorrectPredictionsGenerative(SPost,LTE):
    prediction = numpy.argmax(SPost,axis=0)
    bool_val = numpy.array(prediction==LTE)
    n_samples = prediction.size
    acc = numpy.sum(bool_val)/n_samples
    err = 1 - acc
    return numpy.sum(bool_val)

def computeNaiveSw(DTR,LTR):
    data_list = main.getClassMatrix(DTR,LTR)
    Sw = 0
    for i in range(0,constants.NUM_CLASSES):
        _,CVi = main.computeMeanCovMatrix(data_list[i])
        CVi = CVi * numpy.identity(data_list[i].shape[0])
        Sw += data_list[i].shape[1] * CVi 

    Sw = Sw/DTR.shape[1]
    
    return Sw

def logpdf_GAU_ND(X, mu, C): 
    P = numpy.linalg.inv(C)  
    const = -0.5 * X.shape[0] * numpy.log(2*numpy.pi) 
    const += -0.5 * numpy.linalg.slogdet(C)[1] 
    
    Y = []
    for i in range(X.shape[1]):
        x = X[:, i:i+1]
        diff = x - mu.reshape((X.shape[0],1)) #WITHOUT RESHAPING THE MEAN FROM (X.shape[0], ) TO (X.shape[0],1) WE GET AN ERROR X.shape[0] => N° of Features
        res = const + -0.5 * numpy.dot( diff.T, numpy.dot(P, diff))
        Y.append(res)
        
    return numpy.array(Y).ravel()  

def computeMLestimates(DTR, LTR):
    DP0,DP1 = main.getClassMatrix(DTR,LTR)

    dataset_list = [DP0,DP1]
    mu_list = []
    cov_list = []
    for i in range(0,constants.NUM_CLASSES):
        mu, C = main.computeMeanCovMatrix(dataset_list[i])
        mu_list.append(mu)
        cov_list.append(C)
    
    return mu_list, cov_list

def computeNaiveMLestimates(DTR, LTR):
    DP0,DP1 = main.getClassMatrix(DTR,LTR)

    dataset_list = [DP0,DP1]
    mu_list = []
    cov_list = []
    for i in range(0,constants.NUM_CLASSES):
        mu, C = main.computeMeanCovMatrix(dataset_list[i])
        mu_list.append(mu)
        cov_list.append(C*numpy.identity(dataset_list[i].shape[0]))
    
    return mu_list, cov_list

#MODELS:
def MVG_log_classifier(DTR, LTR, DTE, LTE):
    classDepMU, classDepCOV = computeMLestimates(DTR,LTR)
    S = numpy.zeros((constants.NUM_CLASSES, DTE.shape[1]))

    for i in range(0,constants.NUM_CLASSES):
        for j, sample in enumerate(DTE.T): 
            sample = main.vcol(sample)
            S[i, j] = logpdf_GAU_ND(sample, classDepMU[i], classDepCOV[i]) 
    
    logSJoint = numpy.log(1/constants.NUM_CLASSES) + S 
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0)
    logSPost = logSJoint - logSMarginal

    # return number of correct predictions
    return computeNumCorrectPredictionsGenerative(logSPost, LTE)



def NaiveBayesGaussianClassifier_log(DTR, LTR, DTE, LTE):
    classDepMU, classDepCOV = computeNaiveMLestimates(DTR, LTR)
    S = numpy.zeros((constants.NUM_CLASSES, DTE.shape[1]))

    for i in range(0,constants.NUM_CLASSES):
        for j, sample in enumerate(DTE.T):  
            sample = main.vcol(sample)
            S[i, j] = logpdf_GAU_ND(sample, classDepMU[i], classDepCOV[i]) 
    
    logSJoint = numpy.log(1/constants.NUM_CLASSES) + S 
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0) 
    logSPost = logSJoint - logSMarginal

    # return number of correct predictions
    return computeNumCorrectPredictionsGenerative(logSPost, LTE)
    


def TiedCovarianceGaussianClassifier_log(DTR, LTR, DTE, LTE):
    classDepMU, classDepCOV = computeMLestimates(DTR,LTR)
    Sw = lda.computeSw(DTR,LTR)
    S = numpy.zeros((constants.NUM_CLASSES, DTE.shape[1]))

    for i in range(0,constants.NUM_CLASSES):
        for j, sample in enumerate(DTE.T):  
            sample = main.vcol(sample)
            S[i, j] = logpdf_GAU_ND(sample, classDepMU[i], Sw)  
    
    logSJoint = numpy.log(1/constants.NUM_CLASSES) + S  
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0) 
    logSPost = logSJoint - logSMarginal 
   
    # return number of correct predictions
    return computeNumCorrectPredictionsGenerative(logSPost, LTE)


def TiedNaiveBayesGaussianClassifier_log(DTR, LTR, DTE, LTE):
    classDepMU, classDepCOV = computeNaiveMLestimates(DTR, LTR)
    Sw = computeNaiveSw(DTR,LTR)
    S = numpy.zeros((constants.NUM_CLASSES, DTE.shape[1]))

    for i in range(0,constants.NUM_CLASSES):
        for j, sample in enumerate(DTE.T):  
            sample = main.vcol(sample)
            S[i, j] = logpdf_GAU_ND(sample, classDepMU[i], Sw)  
    
    logSJoint = numpy.log(1/constants.NUM_CLASSES) + S  
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0) 
    logSPost = logSJoint - logSMarginal 
   
    # return number of correct predictions
    return computeNumCorrectPredictionsGenerative(logSPost, LTE)
