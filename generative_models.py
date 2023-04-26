import numpy
import scipy
import scipy.linalg
import main
import math

def computeNumCorrectPredictions(SPost,LTE):
    prediction = numpy.argmax(SPost,axis=0)
    bool_val = numpy.array(prediction==LTE)
    n_samples = prediction.size
    acc = numpy.sum(bool_val)/n_samples
    err = 1 - acc
    print(acc)
    print(err)
    return numpy.sum(bool_val)

def logpdf_GAU_ND(X, mu, C): 
    # compute logN for each xi (x)
    M = X.shape[0]
    N = X.shape[1]
    Y = numpy.empty((1,N))

    for i in range(N):
        x = X[:, i:i+1]
        firstTerm = -M/2 * numpy.log(2 * math.pi)
        _,mod = numpy.linalg.slogdet(C)
        secondTerm = -1/2 * mod
        diff = x - mu.reshape((10,1))
        thirdTerm = -1/2 * numpy.dot(diff.T,numpy.dot(numpy.linalg.inv(C),diff))
        Y[:,i] = firstTerm + secondTerm + thirdTerm

    return Y.ravel()

def computeMLestimates(DTR,LTR):
    DP0,DP1 = main.getClassMatrix(DTR,LTR)

    dataset_list = [DP0,DP1]
    num_classes = 2
    mu_list = []
    cov_list = []
    for i in range(0,num_classes):
        mu,C = main.computeMeanCovMatrix(dataset_list[i])
        #print(mu)
        #print(C)
        mu_list.append(mu)
        cov_list.append(C)
    
    return mu_list,cov_list

def MVG_log_classifier(DTR,LTR,DTE,LTE):
    mu_list,cov_list = computeMLestimates(DTR,LTR)
    # Classification
    # prior = vector of prior probabilities (1/2 since we have 2 classes)     S = Score Matrix    
    prior = main.vcol(numpy.log(numpy.ones(2)/2.0))
    likelihood_array = []
    num_classes = 2
    for i in range(0,num_classes):
        # from the parameter of the class, compute the log density with the test set
        likelihood = logpdf_GAU_ND(DTE,mu_list[i],cov_list[i])
        likelihood = main.vrow(likelihood)
        #print(likelihood)
        likelihood_array.append(likelihood)
    S = numpy.vstack(likelihood_array)
    # SJoint = Matrix of joint densities    SMarginal = Marginal Densities    SPost = array of class posterior probabilities
    logSJoint = S + prior
    logSMarginal = main.vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    return computeNumCorrectPredictions(logSPost,LTE)

'''
def MVG_log_classifier(DTR, LTR, DTE, LTE):
    classDepMU = []
    classDepCOV = []
    num_classes = len(set(LTR))
    for i in range(0,num_classes):
        mu,C = main.computeMeanCovMatrix(DTR[:, LTR==i])
        classDepMU.append(mu) 
        classDepCOV.append(C)
    
    #S = numpy.zeros((num_classes, DTE.shape[1]))
    likelihood_array = []

    for i in range(0,num_classes):
        for j, sample in enumerate(DTE.T): 
            sample = main.vcol(sample)
            likelihood = main.vrow(logpdf_GAU_ND(sample, classDepMU[i], classDepCOV[i]))  
            likelihood_array.append(likelihood)
    
    S = numpy.vstack(likelihood_array)
    logSJoint = numpy.log(1/3) + S  
    logSMarginal = scipy.special.logsumexp(logSJoint, axis=0) 
    logSPost = logSJoint - logSMarginal

    # return number of correct predictions
    return computeNumCorrectPredictions(logSPost,LTE)
'''