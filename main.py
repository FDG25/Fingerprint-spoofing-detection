import numpy
import scipy
import matplotlib.pyplot as plt
import svm
import pca
import lda
import generative_models
import constants
import plot
import lr
import optimal_decision
from plot_utility import PlotUtility

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

def randomize(DTR,LTR):
    numpy.random.seed(0) 
    indexes = numpy.random.permutation(DTR.shape[1])
    DTR_RAND = numpy.zeros((constants.NUM_FEATURES, DTR.shape[1]))
    LTR_RAND = numpy.zeros((LTR.size,))
    index = 0
    for rand_index in indexes:
        DTR_RAND[:,index] = DTR[:,rand_index]
        LTR_RAND[index] = LTR[rand_index]
        index+=1
    return DTR_RAND,LTR_RAND

def randomize_weighted(DTR,LTR):
    # GET TWO DATASET FOR EACH CLASS
    DT0,DT1 = getClassMatrix(DTR,LTR)
    # RANDOMIZE DT0 AND DT1
    numpy.random.seed(0) 
    indexes = numpy.random.permutation(DT0.shape[1])
    DT0_RAND = numpy.zeros((constants.NUM_FEATURES, DT0.shape[1]))
    index = 0
    for rand_index in indexes:
        DT0_RAND[:,index] = DT0[:,rand_index]
        index+=1
    indexes = numpy.random.permutation(DT1.shape[1])
    DT1_RAND = numpy.zeros((constants.NUM_FEATURES, DT1.shape[1]))
    index = 0
    for rand_index in indexes:
        DT1_RAND[:,index] = DT1[:,rand_index]
        index+=1
    # PUT ALL TOGETHER IN THE FINAL RANDOMIZED DATASET
    DTR_RAND = numpy.zeros((constants.NUM_FEATURES, DTR.shape[1]))
    LTR_RAND = numpy.zeros((LTR.size,))
    index_0 = 0
    index_1 = 0
    i = 0
    while i < DTR.shape[1]:
        if i <= 2172:
            DTR_RAND[:,i] = DT0_RAND[:,index_0]
            LTR_RAND[i] = 0
            DTR_RAND[:,i+1] = DT0_RAND[:,index_0+1]
            LTR_RAND[i+1] = 0
            DTR_RAND[:,i+2] = DT1_RAND[:,index_1]
            LTR_RAND[i+2] = 1
            i+=3
            index_0+=2
            index_1+=1
        else:
            DTR_RAND[:,i] = DT0_RAND[:,index_0]
            LTR_RAND[i] = 0
            DTR_RAND[:,i+1] = DT1_RAND[:,index_1]
            LTR_RAND[i+1] = 1
            i+=2
            index_0+=1
            index_1+=1
    return DTR_RAND,LTR_RAND

def K_Fold_Generative(D,L,K):
    # Leave-One-Out Approach Con K=2325: 
    fold_dimension = int(D.shape[1]/K)  # size of each fold
    fold_indices = numpy.arange(0, K*fold_dimension, fold_dimension)  # indices to split the data into folds
    classifiers = [(generative_models.MVG_log_classifier, "Multivariate Gaussian Classifier"), (generative_models.NaiveBayesGaussianClassifier_log, "Naive Bayes"), (generative_models.TiedCovarianceGaussianClassifier_log, "Tied Covariance"), (generative_models.TiedNaiveBayesGaussianClassifier_log, "Tied Naive Bayes")] 
    for classifier_function, classifier_name in classifiers: 
        nWrongPrediction = 0
        scores = numpy.array([])
        labels = numpy.array([])
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
            DTR,P = pca.PCA_projection(DTR,m = constants.M)
            DVA = numpy.dot(P.T, DVA)
            nSamples = DVA.shape[1]  
            scores_i,nCorrectPrediction = classifier_function(DTR, LTR, DVA, LVA) 
            nWrongPrediction += nSamples - nCorrectPrediction
            scores = numpy.append(scores,scores_i)
            labels = numpy.append(labels,LVA)
        errorRate = nWrongPrediction/D.shape[1] 
        accuracy = 1 - errorRate
        print(f"{classifier_name} results:\nAccuracy: {round(accuracy*100, 2)}%\nError rate: {round(errorRate*100, 2)}%\n",end="")
        print(f"Min DCF for {classifier_name}: {optimal_decision.computeMinDCF(constants.PRIOR_PROBABILITY,constants.CFN,constants.CFP,scores,labels)}\n") 

def K_Fold_LR(D,L,K,classifiers,hyperParameter):
    # Leave-One-Out Approach Con K=2325: 
    fold_dimension = int(D.shape[1]/K)  # size of each fold
    fold_indices = numpy.arange(0, K*fold_dimension, fold_dimension)  # indices to split the data into folds
    minDcfs = []
    for classifier_function, classifier_name in classifiers: 
        nWrongPrediction = 0
        scores = numpy.array([])
        labels = numpy.array([])
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
            #DTR,P = pca.PCA_projection(DTR,m = constants.M)
            #DVA = numpy.dot(P.T, DVA)
            nSamples = DVA.shape[1]  
            scores_i,nCorrectPrediction = classifier_function(DTR, LTR, DVA, LVA, hyperParameter) 
            nWrongPrediction += nSamples - nCorrectPrediction
            scores = numpy.append(scores,scores_i)
            labels = numpy.append(labels,LVA)
        errorRate = nWrongPrediction/D.shape[1] 
        accuracy = 1 - errorRate
        print(f"{classifier_name} results:\nAccuracy: {round(accuracy*100, 2)}%\nError rate: {round(errorRate*100, 2)}%\n",end="")
        minDcf = optimal_decision.computeMinDCF(constants.PRIOR_PROBABILITY,constants.CFN,constants.CFP,scores,labels)
        minDcfs.append(minDcf)
        print(f"Min DCF for {classifier_name}: {minDcf}\n")
    return minDcfs 

def K_Fold_SVM_linear(D,L,K,hyperParameter_K,hyperParameter_C):
    # Leave-One-Out Approach Con K=2325: 
    fold_dimension = int(D.shape[1]/K)  # size of each fold
    fold_indices = numpy.arange(0, K*fold_dimension, fold_dimension)  # indices to split the data into folds
    minDcfs = []
    nWrongPrediction = 0
    scores = numpy.array([])
    labels = numpy.array([])
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
        #DTR,P = pca.PCA_projection(DTR,m = constants.M)
        #DVA = numpy.dot(P.T, DVA)
        nSamples = DVA.shape[1]  
        scores_i,nCorrectPrediction = svm.linear_svm(DTR, LTR, DVA, LVA, hyperParameter_K,hyperParameter_C) 
        nWrongPrediction += nSamples - nCorrectPrediction
        scores = numpy.append(scores,scores_i)
        labels = numpy.append(labels,LVA)
    errorRate = nWrongPrediction/D.shape[1] 
    accuracy = 1 - errorRate
    print(f"Linear SVM results:\nAccuracy: {round(accuracy*100, 2)}%\nError rate: {round(errorRate*100, 2)}%\n",end="")
    minDcf = optimal_decision.computeMinDCF(constants.PRIOR_PROBABILITY,constants.CFN,constants.CFP,scores,labels)
    minDcfs.append(minDcf)
    print(f"Min DCF for Linear SVM: {minDcf}\n")
    return minDcfs

def K_Fold_SVM_kernel_polynomial(D,L,K,hyperParameter_K,hyperParameter_C,hyperParameter_c,hyperParameter_d):
    # Leave-One-Out Approach Con K=2325: 
    fold_dimension = int(D.shape[1]/K)  # size of each fold
    fold_indices = numpy.arange(0, K*fold_dimension, fold_dimension)  # indices to split the data into folds
    minDcfs = []
    nWrongPrediction = 0
    scores = numpy.array([])
    labels = numpy.array([])
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
        #DTR,P = pca.PCA_projection(DTR,m = constants.M)
        #DVA = numpy.dot(P.T, DVA)
        nSamples = DVA.shape[1]  
        scores_i,nCorrectPrediction = svm.kernel_svm_polynomial(DTR, LTR, DVA, LVA, hyperParameter_K,hyperParameter_C,hyperParameter_c,hyperParameter_d) 
        nWrongPrediction += nSamples - nCorrectPrediction
        #print("Correct: " + str(nCorrectPrediction))
        #print("Wrong: " + str(nWrongPrediction))
        scores = numpy.append(scores,scores_i)
        labels = numpy.append(labels,LVA)
    errorRate = nWrongPrediction/D.shape[1] 
    accuracy = 1 - errorRate
    print(f"Polynomial Kernel SVM results:\nAccuracy: {round(accuracy*100, 2)}%\nError rate: {round(errorRate*100, 2)}%\n",end="")
    minDcf = optimal_decision.computeMinDCF(constants.PRIOR_PROBABILITY,constants.CFN,constants.CFP,scores,labels)
    minDcfs.append(minDcf)
    print(f"Min DCF for Polynomial Kernel SVM: {minDcf}\n")
    return minDcfs

def K_Fold_SVM_kernel_rbf(D,L,K,hyperParameter_K,hyperParameter_C,hyperParameter_gamma):
    # Leave-One-Out Approach Con K=2325: 
    fold_dimension = int(D.shape[1]/K)  # size of each fold
    fold_indices = numpy.arange(0, K*fold_dimension, fold_dimension)  # indices to split the data into folds
    minDcfs = []
    nWrongPrediction = 0
    scores = numpy.array([])
    labels = numpy.array([])
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
        #DTR,P = pca.PCA_projection(DTR,m = constants.M)
        #DVA = numpy.dot(P.T, DVA)
        nSamples = DVA.shape[1]  
        scores_i,nCorrectPrediction = svm.kernel_svm_radial(DTR, LTR, DVA, LVA, hyperParameter_K,hyperParameter_C,hyperParameter_gamma) 
        nWrongPrediction += nSamples - nCorrectPrediction
        scores = numpy.append(scores,scores_i)
        labels = numpy.append(labels,LVA)
    errorRate = nWrongPrediction/D.shape[1] 
    accuracy = 1 - errorRate
    print(f"RADIAL BASIS FUNCTION (RBF) Kernel SVM results:\nAccuracy: {round(accuracy*100, 2)}%\nError rate: {round(errorRate*100, 2)}%\n",end="")
    minDcf = optimal_decision.computeMinDCF(constants.PRIOR_PROBABILITY,constants.CFN,constants.CFP,scores,labels)
    minDcfs.append(minDcf)
    print(f"Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: {minDcf}\n")
    return minDcfs

def optimalDecision(DTR,LTR,DTE,LTE):
    classifiers = [(generative_models.MVG_log_classifier, "Multivariate Gaussian Classifier"), (generative_models.NaiveBayesGaussianClassifier_log, "Naive Bayes"), (generative_models.TiedCovarianceGaussianClassifier_log, "Tied Covariance"), (generative_models.TiedNaiveBayesGaussianClassifier_log, "Tied Naive Bayes"),(lr.LogisticRegressionWeighted, "Logistic Regression Weighted"),(lr.LogisticRegressionWeightedQuadratic, "Logistic Regression Quadratic Weighted")] 
    for classifier_function, classifier_name in classifiers:
        LP,_ = classifier_function(DTR,LTR,DTE,LTE)
        # se sorto i log likelihoods mi dà problemi tutte le minDCF a 0
        LP = numpy.sort(LP)
        print(f"Min DCF for {classifier_name}: {optimal_decision.computeMinDCF(constants.PRIOR_PROBABILITY,constants.CFN,constants.CFP,LP,LTE)}")

def lr_lambda_parameter_testing(DTR,LTR,lambda_values,classifier):    
    #priors = [0.5, 0.9, 0.1]
    priors = [constants.PRIOR_PROBABILITY]
    minDcfs=[]
    for i in range(len(priors)):
        print("prior:",priors[i])        
        for lambd in lambda_values:
            print("lambda value : " + str(lambd))
            minDcf = K_Fold_LR(DTR,LTR,K=5,classifiers=classifier,hyperParameter=lambd)
            minDcfs.append(minDcf)
    # PLOT
    plot.plotDCF(lambda_values,minDcfs,'lambda')
    
def svm_linear_K_C_parameters_testing(DTR,LTR,k_values,C_values):  
    priors = [constants.PRIOR_PROBABILITY]
    plotUtility_list=[]
    for prior in priors:
        print("prior:",prior)        
        for k_value in k_values:
            print("k value : " + str(k_value))
            for C in C_values:
                print("C value : " + str(C))
                minDcf = K_Fold_SVM_linear(DTR,LTR,K=5,hyperParameter_K=k_value,hyperParameter_C=C)
                plotUtility_list.append(PlotUtility(prior=prior,k=k_value,C=C,minDcf=minDcf))
    
    # ----  SINGLE PLOT FOR K = 1,10,100  -----
    k_1 = list(filter(lambda PlotElement: PlotElement.is_k(1), plotUtility_list))
    minDcfs_k_1 = [PlotElement.getminDcf() for PlotElement in k_1]
    C_values_1 = [PlotElement.getC() for PlotElement in k_1]

    k_10 = list(filter(lambda PlotElement: PlotElement.is_k(10), plotUtility_list))
    minDcfs_k_10 = [PlotElement.getminDcf() for PlotElement in k_10]
    C_values_10 = [PlotElement.getC() for PlotElement in k_10]

    k_100 = list(filter(lambda PlotElement: PlotElement.is_k(100), plotUtility_list))
    minDcfs_k_100 = [PlotElement.getminDcf() for PlotElement in k_100]
    C_values_100 = [PlotElement.getC() for PlotElement in k_100]

    labels = ['K = 1','K = 10','K = 100']
    colors = ['b','g','y']
    #base colors: r, g, b, m, y, c, k, w
    plot.plotDCF([C_values_1,C_values_10,C_values_100],[minDcfs_k_1,minDcfs_k_10,minDcfs_k_100],labels,colors,'C')

def svm_kernel_polynomial_K_C_c_d_parameter_testing(DTR,LTR,k_values,C_values,c_values,d_values):
    priors = [constants.PRIOR_PROBABILITY]
    plotUtility_list=[]
    for prior in priors:
        print("prior:",prior)        
        for k_value in k_values:
            print("k value : " + str(k_value))
            for C in C_values:
                print("C value : " + str(C))
                for c in c_values:
                    print("c value : " + str(c))
                    for d in d_values:
                        print("d value : " + str(d))
                        minDcf = K_Fold_SVM_kernel_polynomial(DTR,LTR,K=5,hyperParameter_K=k_value,hyperParameter_C=C,hyperParameter_c=c,hyperParameter_d=d)
                        plotUtility_list.append(PlotUtility(prior=prior,k=k_value,C=C,c=c,d=d,minDcf=minDcf))
    
    # ----- PLOT FOR K = 1 ------
    k_1_c_0_d_2 = list(filter(lambda PlotElement: PlotElement.is_k(1) and PlotElement.is_c(0) and PlotElement.is_d(2), plotUtility_list))
    minDcfs_k_1_c_0_d_2 = [PlotElement.getminDcf() for PlotElement in k_1_c_0_d_2]
    C_values_k_1_c_0_d_2 = [PlotElement.getC() for PlotElement in k_1_c_0_d_2]

    k_1_c_0_d_4 = list(filter(lambda PlotElement: PlotElement.is_k(1) and PlotElement.is_c(0) and PlotElement.is_d(4), plotUtility_list))
    minDcfs_k_1_c_0_d_4 = [PlotElement.getminDcf() for PlotElement in k_1_c_0_d_4]
    C_values_k_1_c_0_d_4 = [PlotElement.getC() for PlotElement in k_1_c_0_d_4]

    k_1_c_1_d_2 = list(filter(lambda PlotElement: PlotElement.is_k(1) and PlotElement.is_c(1) and PlotElement.is_d(2), plotUtility_list))
    minDcfs_k_1_c_1_d_2 = [PlotElement.getminDcf() for PlotElement in k_1_c_1_d_2]
    C_values_k_1_c_1_d_2 = [PlotElement.getC() for PlotElement in k_1_c_1_d_2]

    k_1_c_1_d_4 = list(filter(lambda PlotElement: PlotElement.is_k(1) and PlotElement.is_c(1) and PlotElement.is_d(4), plotUtility_list))
    minDcfs_k_1_c_1_d_4 = [PlotElement.getminDcf() for PlotElement in k_1_c_1_d_4]
    C_values_k_1_c_1_d_4 = [PlotElement.getC() for PlotElement in k_1_c_1_d_4]

    labels = ['K = 1,c = 0,D = 2','K = 1,c = 0,D = 4','K = 1,c = 1,D = 2','K = 1,c = 1,D = 4']
    colors = ['b','g','y','c']
    #base colors: r, g, b, m, y, c, k, w
    plot.plotDCF([C_values_k_1_c_0_d_2,C_values_k_1_c_0_d_4,C_values_k_1_c_1_d_2,C_values_k_1_c_1_d_4],[minDcfs_k_1_c_0_d_2,minDcfs_k_1_c_0_d_4,minDcfs_k_1_c_1_d_2,minDcfs_k_1_c_1_d_4],labels,colors,'C')


    # ----- PLOT FOR K = 10 ------
    k_10_c_0_d_2 = list(filter(lambda PlotElement: PlotElement.is_k(10) and PlotElement.is_c(0) and PlotElement.is_d(2), plotUtility_list))
    minDcfs_k_10_c_0_d_2 = [PlotElement.getminDcf() for PlotElement in k_10_c_0_d_2]
    C_values_k_10_c_0_d_2 = [PlotElement.getC() for PlotElement in k_10_c_0_d_2]

    k_10_c_0_d_4 = list(filter(lambda PlotElement: PlotElement.is_k(10) and PlotElement.is_c(0) and PlotElement.is_d(4), plotUtility_list))
    minDcfs_k_10_c_0_d_4 = [PlotElement.getminDcf() for PlotElement in k_10_c_0_d_4]
    C_values_k_10_c_0_d_4 = [PlotElement.getC() for PlotElement in k_10_c_0_d_4]

    k_10_c_1_d_2 = list(filter(lambda PlotElement: PlotElement.is_k(10) and PlotElement.is_c(1) and PlotElement.is_d(2), plotUtility_list))
    minDcfs_k_10_c_1_d_2 = [PlotElement.getminDcf() for PlotElement in k_10_c_1_d_2]
    C_values_k_10_c_1_d_2 = [PlotElement.getC() for PlotElement in k_10_c_1_d_2]

    k_10_c_1_d_4 = list(filter(lambda PlotElement: PlotElement.is_k(10) and PlotElement.is_c(1) and PlotElement.is_d(4), plotUtility_list))
    minDcfs_k_10_c_1_d_4 = [PlotElement.getminDcf() for PlotElement in k_10_c_1_d_4]
    C_values_k_10_c_1_d_4 = [PlotElement.getC() for PlotElement in k_10_c_1_d_4]

    labels = ['K = 10,c = 0,D = 2','K = 10,c = 0,D = 4','K = 10,c = 1,D = 2','K = 10,c = 1,D = 4']
    colors = ['b','g','y','c']
    #base colors: r, g, b, m, y, c, k, w
    plot.plotDCF([C_values_k_10_c_0_d_2,C_values_k_10_c_0_d_4,C_values_k_10_c_1_d_2,C_values_k_10_c_1_d_4],[minDcfs_k_10_c_0_d_2,minDcfs_k_10_c_0_d_4,minDcfs_k_10_c_1_d_2,minDcfs_k_10_c_1_d_4],labels,colors,'C')

    
    # ----- PLOT FOR K = 100 ------
    k_100_c_0_d_2 = list(filter(lambda PlotElement: PlotElement.is_k(100) and PlotElement.is_c(0) and PlotElement.is_d(2), plotUtility_list))
    minDcfs_k_100_c_0_d_2 = [PlotElement.getminDcf() for PlotElement in k_100_c_0_d_2]
    C_values_k_100_c_0_d_2 = [PlotElement.getC() for PlotElement in k_100_c_0_d_2]

    k_100_c_0_d_4 = list(filter(lambda PlotElement: PlotElement.is_k(100) and PlotElement.is_c(0) and PlotElement.is_d(4), plotUtility_list))
    minDcfs_k_100_c_0_d_4 = [PlotElement.getminDcf() for PlotElement in k_100_c_0_d_4]
    C_values_k_100_c_0_d_4 = [PlotElement.getC() for PlotElement in k_100_c_0_d_4]

    k_100_c_1_d_2 = list(filter(lambda PlotElement: PlotElement.is_k(100) and PlotElement.is_c(1) and PlotElement.is_d(2), plotUtility_list))
    minDcfs_k_100_c_1_d_2 = [PlotElement.getminDcf() for PlotElement in k_100_c_1_d_2]
    C_values_k_100_c_1_d_2 = [PlotElement.getC() for PlotElement in k_100_c_1_d_2]

    k_100_c_1_d_4 = list(filter(lambda PlotElement: PlotElement.is_k(100) and PlotElement.is_c(1) and PlotElement.is_d(4), plotUtility_list))
    minDcfs_k_100_c_1_d_4 = [PlotElement.getminDcf() for PlotElement in k_100_c_1_d_4]
    C_values_k_100_c_1_d_4 = [PlotElement.getC() for PlotElement in k_100_c_1_d_4]

    labels = ['K = 100,c = 0,D = 2','K = 100,c = 0,D = 4','K = 100,c = 1,D = 2','K = 100,c = 1,D = 4']
    colors = ['b','g','y','c']
    #base colors: r, g, b, m, y, c, k, w
    plot.plotDCF([C_values_k_100_c_0_d_2,C_values_k_100_c_0_d_4,C_values_k_100_c_1_d_2,C_values_k_100_c_1_d_4],[minDcfs_k_100_c_0_d_2,minDcfs_k_100_c_0_d_4,minDcfs_k_100_c_1_d_2,minDcfs_k_100_c_1_d_4],labels,colors,'C')

def svm_kernel_rbf_K_C_gamma_parameter_testing(DTR,LTR,k_values,C_values,gamma_values):
    priors = [constants.PRIOR_PROBABILITY]
    plotUtility_list=[]
    for prior in priors:
        print("prior:",prior)        
        for k_value in k_values:
            print("k value : " + str(k_value))
            for C in C_values:
                print("C value : " + str(C))
                for gamma in gamma_values:
                    print("gamma value : " + str(gamma))
                    minDcf = K_Fold_SVM_kernel_rbf(DTR,LTR,K=5,hyperParameter_K=k_value,hyperParameter_C=C,hyperParameter_gamma=gamma)
                    plotUtility_list.append(PlotUtility(prior=prior,k=k_value,C=C,gamma=gamma,minDcf=minDcf))
    
    # ---- PLOT FOR K = 0 -----
    k_0_gamma_1e1 = list(filter(lambda PlotElement: PlotElement.is_k(0) and PlotElement.is_gamma(1/numpy.exp(1)), plotUtility_list))
    minDcfs_k_0_gamma_1e1 = [PlotElement.getminDcf() for PlotElement in k_0_gamma_1e1]
    C_values_k_0_gamma_1e1 = [PlotElement.getC() for PlotElement in k_0_gamma_1e1]

    k_0_gamma_1e2 = list(filter(lambda PlotElement: PlotElement.is_k(0) and PlotElement.is_gamma(1/numpy.exp(2)), plotUtility_list))
    minDcfs_k_0_gamma_1e2 = [PlotElement.getminDcf() for PlotElement in k_0_gamma_1e2]
    C_values_k_0_gamma_1e2 = [PlotElement.getC() for PlotElement in k_0_gamma_1e2]

    k_0_gamma_1e3 = list(filter(lambda PlotElement: PlotElement.is_k(0) and PlotElement.is_gamma(1/numpy.exp(3)), plotUtility_list))
    minDcfs_k_0_gamma_1e3 = [PlotElement.getminDcf() for PlotElement in k_0_gamma_1e3]
    C_values_k_0_gamma_1e3 = [PlotElement.getC() for PlotElement in k_0_gamma_1e3]

    k_0_gamma_1e4 = list(filter(lambda PlotElement: PlotElement.is_k(0) and PlotElement.is_gamma(1/numpy.exp(4)), plotUtility_list))
    minDcfs_k_0_gamma_1e4 = [PlotElement.getminDcf() for PlotElement in k_0_gamma_1e4]
    C_values_k_0_gamma_1e4 = [PlotElement.getC() for PlotElement in k_0_gamma_1e4]

    k_0_gamma_1e5 = list(filter(lambda PlotElement: PlotElement.is_k(0) and PlotElement.is_gamma(1/numpy.exp(5)), plotUtility_list))
    minDcfs_k_0_gamma_1e5 = [PlotElement.getminDcf() for PlotElement in k_0_gamma_1e5]
    C_values_k_0_gamma_1e5 = [PlotElement.getC() for PlotElement in k_0_gamma_1e5]

    labels = ['K = 0,log(γ) = -1','K = 0,log(γ) = -2','K = 0,log(γ) = -3','K = 0,log(γ) = -4','K = 0,log(γ) = -5']
    colors = ['b','g','y','c','r']
    #base colors: r, g, b, m, y, c, k, w
    plot.plotDCF([C_values_k_0_gamma_1e1,C_values_k_0_gamma_1e2,C_values_k_0_gamma_1e3,C_values_k_0_gamma_1e4,C_values_k_0_gamma_1e5],[minDcfs_k_0_gamma_1e1,minDcfs_k_0_gamma_1e2,minDcfs_k_0_gamma_1e3,minDcfs_k_0_gamma_1e4,minDcfs_k_0_gamma_1e5],labels,colors,'C')


    # ---- PLOT FOR K = 1 -----
    k_1_gamma_1e1 = list(filter(lambda PlotElement: PlotElement.is_k(1) and PlotElement.is_gamma(1/numpy.exp(1)), plotUtility_list))
    minDcfs_k_1_gamma_1e1 = [PlotElement.getminDcf() for PlotElement in k_1_gamma_1e1]
    C_values_k_1_gamma_1e1 = [PlotElement.getC() for PlotElement in k_1_gamma_1e1]

    k_1_gamma_1e2 = list(filter(lambda PlotElement: PlotElement.is_k(1) and PlotElement.is_gamma(1/numpy.exp(2)), plotUtility_list))
    minDcfs_k_1_gamma_1e2 = [PlotElement.getminDcf() for PlotElement in k_1_gamma_1e2]
    C_values_k_1_gamma_1e2 = [PlotElement.getC() for PlotElement in k_1_gamma_1e2]

    k_1_gamma_1e3 = list(filter(lambda PlotElement: PlotElement.is_k(1) and PlotElement.is_gamma(1/numpy.exp(3)), plotUtility_list))
    minDcfs_k_1_gamma_1e3 = [PlotElement.getminDcf() for PlotElement in k_1_gamma_1e3]
    C_values_k_1_gamma_1e3 = [PlotElement.getC() for PlotElement in k_1_gamma_1e3]

    k_1_gamma_1e4 = list(filter(lambda PlotElement: PlotElement.is_k(1) and PlotElement.is_gamma(1/numpy.exp(4)), plotUtility_list))
    minDcfs_k_1_gamma_1e4 = [PlotElement.getminDcf() for PlotElement in k_1_gamma_1e4]
    C_values_k_1_gamma_1e4 = [PlotElement.getC() for PlotElement in k_1_gamma_1e4]

    k_1_gamma_1e5 = list(filter(lambda PlotElement: PlotElement.is_k(1) and PlotElement.is_gamma(1/numpy.exp(5)), plotUtility_list))
    minDcfs_k_1_gamma_1e5 = [PlotElement.getminDcf() for PlotElement in k_1_gamma_1e5]
    C_values_k_1_gamma_1e5 = [PlotElement.getC() for PlotElement in k_1_gamma_1e5]

    labels = ['K = 1,log(γ) = -1','K = 1,log(γ) = -2','K = 1,log(γ) = -3','K = 1,log(γ) = -4','K = 1,log(γ) = -5']
    colors = ['b','g','y','c','r']
    #base colors: r, g, b, m, y, c, k, w
    plot.plotDCF([C_values_k_1_gamma_1e1,C_values_k_1_gamma_1e2,C_values_k_1_gamma_1e3,C_values_k_1_gamma_1e4,C_values_k_1_gamma_1e5],[minDcfs_k_1_gamma_1e1,minDcfs_k_1_gamma_1e2,minDcfs_k_1_gamma_1e3,minDcfs_k_1_gamma_1e4,minDcfs_k_1_gamma_1e5],labels,colors,'C')

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
    #Sw = lda.computeSw(DTR,LTR)
    #Sb = lda.computeSb(DTR,LTR)
    #DTRP = lda.LDA1(m=1,Sb=Sb,Sw=Sw,D=DTR)
    #plot.plot_hist_projected_data_lda(DTRP,LTR)
    #plot.plot_Heatmap_Whole_Dataset(DTR)
    #plot.plot_Heatmap_Spoofed_Authentic(DTR,LTR,Class_Label=0)
    #plot.plot_Heatmap_Spoofed_Authentic(DTR,LTR,Class_Label=1)
    # ---------------   GENERATIVE MODELS   -----------------------
    # MVG_LOG_CLASSIFIER
    # generative_models.MVG_log_classifier(DTR,LTR,DTE,LTE)
    # generative_models.NaiveBayesGaussianClassifier_log(DTR,LTR,DTE,LTE)
    # generative_models.TiedCovarianceGaussianClassifier_log(DTR,LTR,DTE,LTE)
    # generative_models.TiedNaiveBayesGaussianClassifier_log(DTR,LTR,DTE,LTE)
    # RANDOMIZE DATASET BEFORE K-FOLD
    DTR_RAND,LTR_RAND = randomize(DTR,LTR)
    DTE_RAND,LTE_RAND = randomize(DTE,LTE)
    #print("K_Fold with K = 5")
    #print("PCA with m = " + str(constants.M))
    #K_Fold_Generative(DTR_RAND,LTR_RAND,K=5)

    # ---------------   LR MODELS   -----------------------
    # CALL K-FOLD AND TEST THE HYPERPARAMETER
    print("K_Fold with K = 5\n\n")
    #print("PCA with m = " + str(constants.M))
    lambda_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
    classifier = [(lr.LogisticRegressionWeighted, "Logistic Regression Weighted"),(lr.LogisticRegressionWeightedQuadratic, "Logistic Regression Weighted Quadratic")]
    lr_lambda_parameter_testing(DTR_RAND,LTR_RAND,lambda_values,classifier)
    #print("No Weight")
    #lr.LogisticRegressionWeighted(DTR,LTR,DTE,LTE)
    #print("Weight")
    #lr.LogisticRegression(DTR,LTR,DTE,LTE)


    # ---------------   SVM MODELS   -----------------------
    print("SVM LINEAR HYPERPARAMETERS K AND C TESTING:")
    K_values = [1, 10, 100]
    C_values = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    svm_linear_K_C_parameters_testing(DTR_RAND,LTR_RAND,K_values,C_values)
    
    print("SVM POLYNOMIAL K,C,c,d TESTING:")
    K_values = [1, 10, 100]
    C_values = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0] # for C <= 10^-6 there is a significative worsening in performance 
    c_values = [0, 1]
    d_values = [2.0, 4.0]
    svm_kernel_polynomial_K_C_c_d_parameter_testing(DTR_RAND,LTR_RAND,K_values,C_values,c_values,d_values)

    print("SVM RADIAL BASIS FUNCTION (RBF) K,C,gamma TESTING:")
    K_values = [0.0, 1.0]
    C_values = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    # we want log(gamma), so we pass gamma value for which log(gamma) = -1,-2,-3,-4,-5
    gamma_values = [1.0/numpy.exp(1), 1.0/numpy.exp(2), 1.0/numpy.exp(3), 1.0/numpy.exp(4), 1.0/numpy.exp(5)] #hyper-parameter
    svm_kernel_rbf_K_C_gamma_parameter_testing(DTR_RAND,LTR_RAND,K_values,C_values,gamma_values)
    # ------------------ OPTIMAL DECISION --------------------------
    #optimalDecision(DTR_RAND,LTR_RAND,DTE_RAND,LTE_RAND)
    #We now turn our attention at evaluating the predictions made by our classifier R for a target application
    #with prior and costs given by (π1, Cfn, Cfp).
    #LP,_ = generative_models.MVG_log_classifier(DTR_RAND, LTR_RAND, DTE_RAND, LTE_RAND)
    #LP = numpy.sort(LP)
    #print("MVG minDCF: " + str(optimal_decision.computeMinDCF(0.5,1,10,LP,LTE_RAND))) 
    #LP,_ = generative_models.NaiveBayesGaussianClassifier_log(DTR_RAND, LTR_RAND, DTE_RAND, LTE_RAND)
    #àLP = numpy.sort(LP)
    #print("Naive Bayes minDCF: " + str(optimal_decision.computeMinDCF(0.5,1,10,LP,LTE_RAND)))
    #LP,_ = generative_models.TiedCovarianceGaussianClassifier_log(DTR_RAND, LTR_RAND, DTE_RAND, LTE_RAND)
    #LP = numpy.sort(LP)
    #print("Tied minDCF: " + str(optimal_decision.computeMinDCF(0.5,1,10,LP,LTE_RAND))) 
    #LP,_ = generative_models.TiedNaiveBayesGaussianClassifier_log(DTR_RAND, LTR_RAND, DTE_RAND, LTE_RAND)
    #LP = numpy.sort(LP)
    #print("Tied Naive minDCF: " + str(optimal_decision.computeMinDCF(0.5,1,10,LP,LTE_RAND))) 
    #LP,_ = lr.LogisticRegressionWeighted(DTR_RAND, LTR_RAND, DTE_RAND, LTE_RAND)
    #LP = numpy.sort(LP)
    #print("Logistic Regression Weighted minDCF: " + str(optimal_decision.computeMinDCF(0.5,1,10,LP,LTE_RAND))) 
    #LP,_ = lr.LogisticRegressionWeightedQuadratic(DTR_RAND, LTR_RAND, DTE_RAND, LTE_RAND)
    #LP = numpy.sort(LP)
    #print("Logistic Regression Weighted Quadratic minDCF: " + str(optimal_decision.computeMinDCF(0.5,1,10,LP,LTE_RAND))) 

