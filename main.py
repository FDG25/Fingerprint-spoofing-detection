import numpy
import matplotlib.pyplot as plt
import pca
import lda
import generative_models
import constants

#change the shape of an array from horizontal to vertical, so obtain a column vector
def vcol(array):
    return array.reshape((array.size, 1))

#change the shape of an array from horizontal to vertical, so obtain a row vector
def vrow(array):
    return array.reshape((1, array.size))

def computeMeanCovMatrix(DTR):
    mu = DTR.mean(1)
    DC = DTR - vcol(mu)
    C = numpy.dot(DC,DC.T)/DTR.shape[1]
    return mu, C

def getClassMatrix(DP,LTR):
    # 'spoofed-fingerprint' : name = 0 'authentic-fingerprint' : name = 1 
    DP0 = DP[:, LTR==0]   
    DP1 = DP[:, LTR==1]   
    
    return DP0,DP1

def load(fname): 
    DList = [] 
    labelsList = [] 
 
    with open(fname) as f: 
        for line in f: 
            try:  
                attrs = line.split(',')[0:constants.NUM_FEATURES]  
                attrs = vcol(numpy.array([float(i) for i in attrs]))   
                name = line.split(',')[-1].strip()
                # 'spoofed-fingerprint' : name = 0 'authentic-fingerprint' : name = 1 
                label = int(name)
                DList.append(attrs) 
                labelsList.append(label) 
            except: 
                pass 
 
    return numpy.hstack(DList), numpy.array(labelsList, dtype=numpy.int32)



if __name__ == '__main__':
    # DTR = matrix of 10 rows(NUM_FEATURES) times 2325 samples
    # LTR = unidimensional array of 2325 labels, 1 for each sample
    DTR,LTR = load("Train.txt")
    DTE,LTE = load("Test.txt")
    # PCA
    # DP = projected dataset obtained by projecting our original dataset over a m-dimensional subspace
    DP = pca.PCA_projection(DTR,m=8)
    # LDA
    Sw = lda.computeSw(DTR,LTR)
    Sb = lda.computeSb(DTR,LTR)
    DP = lda.LDA1(m=1,Sb=Sb,Sw=Sw,DTR=DTR)
    # ---------------   GENERATIVE MODELS   -----------------------
    # MVG_LOG_CLASSIFIER
    generative_models.MVG_log_classifier(DTR,LTR,DTE,LTE)
    generative_models.NaiveBayesGaussianClassifier_log(DTR,LTR,DTE,LTE)
    generative_models.TiedCovarianceGaussianClassifier_log(DTR,LTR,DTE,LTE)
    generative_models.TiedNaiveBayesGaussianClassifier_log(DTR,LTR,DTE,LTE)
