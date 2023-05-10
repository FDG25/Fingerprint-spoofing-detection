import numpy
import scipy
import matplotlib.pyplot as plt

import pca
import lda
import generative_models
import constants
import plot
import discriminative_models

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

def getClassMatrix(DTRP,LTR):
    # 'spoofed-fingerprint' : name = 0 'authentic-fingerprint' : name = 1 
    DP0 = DTRP[:, LTR==0]   
    DP1 = DTRP[:, LTR==1]   
    
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

def K_Fold(D,L,K):
    # Leave-One-Out Approach Con K=2325: 
    fold_dimension = int(D.shape[1]/K)  # size of each fold
    fold_indices = numpy.arange(0, K*fold_dimension, fold_dimension)  # indices to split the data into folds
    classifiers = [(generative_models.MVG_log_classifier, "Multivariate Gaussian Classifier"), (generative_models.NaiveBayesGaussianClassifier_log, "Naive Bayes"), (generative_models.TiedCovarianceGaussianClassifier_log, "Tied Covariance"), (generative_models.TiedNaiveBayesGaussianClassifier_log, "Tied Naive Bayes"),(discriminative_models.LogisticRegression, "Logistic Regression")] 
 
    for classifier_function, classifier_name in classifiers: 
        nWrongPrediction = 0 
        # Run k-fold cross-validation
        for i in range(K):    
            # Split the data into training and validation sets
            mask = numpy.zeros(D.shape[1], dtype=bool)
            mask[fold_indices[i]:fold_indices[i]+fold_dimension] = True
            DTR = D[:,~mask]
            LTR = L[~mask]
            DVA = D[:,mask]
            LVA = L[mask]
            # apply PCA on current fold DTR,DVA
            #DTR,P = pca.PCA_projection(DTR,m=8)
            #DVA = numpy.dot(P.T, DVA)
            nSamples = DVA.shape[1]  
            nCorrectPrediction = classifier_function(DTR, LTR, DVA, LVA) 
            nWrongPrediction += nSamples - nCorrectPrediction
        errorRate = nWrongPrediction/D.shape[1] 
        accuracy = 1 - errorRate
        print(f"{classifier_name} results:\nAccuracy: {round(accuracy*100, 2)}%\nError rate: {round(errorRate*100, 2)}%\n")  

if __name__ == '__main__':
    # DTR = matrix of 10 rows(NUM_FEATURES) times 2325 samples
    # LTR = unidimensional array of 2325 labels, 1 for each sample
    DTR,LTR = load("Train.txt")
    DTE,LTE = load("Test.txt")
    # ---------------   PLOT BEFORE DIMENSIONALITY REDUCTION   -----------------------
    #plot.plot_hist(DTR,LTR)
    #plot.plot_scatter(DTR,LTR)
    # PCA (NON HA SENSO FARLO PRIMA)
    # DTRP = projected training set obtained by projecting our original training set over a m-dimensional subspace
    # DTEP = projected test set obtained by projecting our original test set over a m-dimensional subspace
    #m = 2
    #DTRP,_ = pca.PCA_projection(DTR,m)
    #plot.plot_scatter_projected_data_pca(DTRP,LTR)
    # LDA
    Sw = lda.computeSw(DTR,LTR)
    Sb = lda.computeSb(DTR,LTR)
    DTRP = lda.LDA1(m=1,Sb=Sb,Sw=Sw,D=DTR)
    #plot.plot_hist_projected_data_lda(DTRP,LTR)
    # ---------------   GENERATIVE MODELS   -----------------------
    # MVG_LOG_CLASSIFIER
    # generative_models.MVG_log_classifier(DTR,LTR,DTE,LTE)
    # generative_models.NaiveBayesGaussianClassifier_log(DTR,LTR,DTE,LTE)
    # generative_models.TiedCovarianceGaussianClassifier_log(DTR,LTR,DTE,LTE)
    # generative_models.TiedNaiveBayesGaussianClassifier_log(DTR,LTR,DTE,LTE)
    # RANDOMIZE DATASET BEFORE K-FOLD
    numpy.random.seed(0) 
    indexes = numpy.random.permutation(DTR.shape[1])
    DTR_RAND = numpy.zeros((constants.NUM_FEATURES, DTR.shape[1]))
    LTR_RAND = numpy.zeros((LTR.size,))
    index = 0
    for rand_index in indexes:
        DTR_RAND[:,index] = DTR[:,rand_index]
        LTR_RAND[index] = LTR[rand_index]
        index+=1

    print("K_Fold with K = 5")
    K_Fold(DTR_RAND,LTR_RAND,K=5)
    print("Leave One Out (K = 2325)")
    K_Fold(DTR_RAND,LTR_RAND,K=2325)
