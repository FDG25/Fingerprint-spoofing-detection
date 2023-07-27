import numpy
import generative_models
import normalization
import pca
import constants
import optimal_decision
import lr
import plot
import svm
import gmm
import main

def K_Fold_Generative(D,L,K,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=None):
    # Leave-One-Out Approach Con K=2325: 
    fold_dimension = int(D.shape[1]/K)  # size of each fold
    fold_indices = numpy.arange(0, K*fold_dimension, fold_dimension)  # indices to split the data into folds
    classifiers = [(generative_models.MVG_log_classifier, "Multivariate Gaussian Classifier"), (generative_models.NaiveBayesGaussianClassifier_log, "Naive Bayes"), (generative_models.TiedCovarianceGaussianClassifier_log, "Tied Covariance"), (generative_models.TiedNaiveBayesGaussianClassifier_log, "Tied Naive Bayes")] 
    #classifiers = [(generative_models.MVG_log_classifier, "Multivariate Gaussian Classifier")]
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
            if Z_Norm_Flag:
                # apply z-normalization
                DTR = normalization.zNormalizingData(DTR)
                DVA = normalization.zNormalizingData(DVA)
            if PCA_Flag and M!=None:
                # apply PCA on current fold DTR,DVA
                DTR,P = pca.PCA_projection(DTR,m = M)
                DVA = numpy.dot(P.T, DVA)
            nSamples = DVA.shape[1]  
            scores_i,nCorrectPrediction = classifier_function(DTR, LTR, DVA, LVA) 
            nWrongPrediction += nSamples - nCorrectPrediction
            scores = numpy.append(scores,scores_i)
            labels = numpy.append(labels,LVA)
        errorRate = nWrongPrediction/D.shape[1] 
        accuracy = 1 - errorRate
        print(f"{classifier_name} results:\nAccuracy: {round(accuracy*100, 2)}%\nError rate: {round(errorRate*100, 2)}%\n",end="")
        print(f"Min DCF for {classifier_name}: {optimal_decision.computeMinDCF(Dcf_Prior,constants.CFN,constants.CFP,scores,labels)}\n") 

def K_Fold_LR(D,L,K,classifiers,lambd,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=None,Calibration_Flag=None):
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
            if Z_Norm_Flag:
                # apply z-normalization
                DTR = normalization.zNormalizingData(DTR)
                DVA = normalization.zNormalizingData(DVA)
            if PCA_Flag and M!=None:
                # apply PCA on current fold DTR,DVA
                DTR,P = pca.PCA_projection(DTR,m = M)
                DVA = numpy.dot(P.T, DVA)
            nSamples = DVA.shape[1]  
            scores_i,nCorrectPrediction = classifier_function(DTR, LTR, DVA, LVA, lambd) 
            nWrongPrediction += nSamples - nCorrectPrediction
            scores = numpy.append(scores,scores_i)
            labels = numpy.append(labels,LVA)
        errorRate = nWrongPrediction/D.shape[1] 
        accuracy = 1 - errorRate
        print(f"{classifier_name} results:\nAccuracy: {round(accuracy*100, 2)}%\nError rate: {round(errorRate*100, 2)}%\n",end="")
        minDcf = optimal_decision.computeMinDCF(Dcf_Prior,constants.CFN,constants.CFP,scores,labels)
        minDcfs.append(minDcf)
        print(f"Min DCF for {classifier_name}: {minDcf}\n")
        if Calibration_Flag:
            # ----- MISCALIBRATED PLOT --------
            plot.compute_bayes_error_plot(scores,labels,"LR")
            # ----- CALIBRATION AND CALIBRATED PLOT ------
            K_Fold_Calibration(scores,labels,K=5,plt_title="Calibrated " + classifier_name)   
    return minDcfs 

def K_Fold_Calibration(D,L,K,plt_title):
    # ------- SHUFFLING OF SCORES AND LABELS ---------
    D = numpy.hstack(D)
    
    numpy.random.seed(100) 
    indexes = numpy.random.permutation(D.shape[0])
    D_rand = numpy.zeros((1, D.shape[0]))
    L_rand = numpy.zeros((L.size,))
    index = 0
    for rand_index in indexes:
        D_rand[0,index] = D[rand_index]
        L_rand[index] = L[rand_index]
        index+=1

    # Leave-One-Out Approach Con K=2325: 
    fold_dimension = int(D_rand.shape[1]/K)  # size of each fold
    fold_indices = numpy.arange(0, K*fold_dimension, fold_dimension)  # indices to split the data into folds
    classifiers = [(lr.LogisticRegressionPriorWeighted, "Prior Weighted")]
    #minDcfs = []
    for classifier_function, classifier_name in classifiers: 
        nWrongPrediction = 0
        scores = numpy.array([])
        labels = numpy.array([])
        # Run k-fold cross-validation
        for i in range(K):    
            # Split the data into training and validation sets
            mask = numpy.zeros(D_rand.shape[1], dtype=bool)
            mask[fold_indices[i]:fold_indices[i]+fold_dimension] = True
            DTR = D_rand[:,~mask]
            LTR = L_rand[~mask]
            DVA = D_rand[:,mask]
            LVA = L_rand[mask]
            # apply PCA on current fold DTR,DVA
            #DTR,P = pca.PCA_projection(DTR,m = constants.M)
            #DVA = numpy.dot(P.T, DVA)
            nSamples = DVA.shape[1]  
            scores_i,nCorrectPrediction = lr.LogisticRegressionPriorWeighted(DTR, LTR, DVA, LVA) 
            nWrongPrediction += nSamples - nCorrectPrediction
            scores = numpy.append(scores,scores_i)
            labels = numpy.append(labels,LVA)
        #errorRate = nWrongPrediction/D.shape[0] 
        #accuracy = 1 - errorRate
        #print(f"{classifier_name} results:\nAccuracy: {round(accuracy*100, 2)}%\nError rate: {round(errorRate*100, 2)}%\n",end="")
        #minDcf = optimal_decision.computeMinDCF(constants.PRIOR_PROBABILITY,constants.CFN,constants.CFP,scores,labels)
        #minDcfs.append(minDcf)
        #print(f"Min DCF for {classifier_name}: {minDcf}\n")
        plot.compute_bayes_error_plot(scores,labels,plt_title)

def K_Fold_SVM_linear(D,L,K,hyperParameter_K,hyperParameter_C,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=None,Calibration_Flag=None):
    # Leave-One-Out Approach Con K=2325: 
    fold_dimension = int(D.shape[1]/K)  # size of each fold
    fold_indices = numpy.arange(0, K*fold_dimension, fold_dimension)  # indices to split the data into folds
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
        if Z_Norm_Flag:
            # apply z-normalization
            DTR = normalization.zNormalizingData(DTR)
            DVA = normalization.zNormalizingData(DVA)
        if PCA_Flag and M!=None:
            # apply PCA on current fold DTR,DVA
            DTR,P = pca.PCA_projection(DTR,m = M)
            DVA = numpy.dot(P.T, DVA)
        nSamples = DVA.shape[1]  
        scores_i,nCorrectPrediction = svm.linear_svm(DTR, LTR, DVA, LVA, hyperParameter_K,hyperParameter_C) 
        nWrongPrediction += nSamples - nCorrectPrediction
        scores = numpy.append(scores,scores_i)
        labels = numpy.append(labels,LVA)
    errorRate = nWrongPrediction/D.shape[1] 
    accuracy = 1 - errorRate
    print(f"Linear SVM results:\nAccuracy: {round(accuracy*100, 2)}%\nError rate: {round(errorRate*100, 2)}%\n",end="")
    minDcf = optimal_decision.computeMinDCF(Dcf_Prior,constants.CFN,constants.CFP,scores,labels)
    print(f"Min DCF for Linear SVM: {minDcf}\n")
    return minDcf

def K_Fold_SVM_kernel_polynomial(D,L,K,hyperParameter_K,hyperParameter_C,hyperParameter_c,hyperParameter_d,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=None,Calibration_Flag=None):
    # Leave-One-Out Approach Con K=2325: 
    fold_dimension = int(D.shape[1]/K)  # size of each fold
    fold_indices = numpy.arange(0, K*fold_dimension, fold_dimension)  # indices to split the data into folds
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
        if Z_Norm_Flag:
            # apply z-normalization
            DTR = normalization.zNormalizingData(DTR)
            DVA = normalization.zNormalizingData(DVA)
        if PCA_Flag and M!=None:
            # apply PCA on current fold DTR,DVA
            DTR,P = pca.PCA_projection(DTR,m = M)
            DVA = numpy.dot(P.T, DVA)
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
    minDcf = optimal_decision.computeMinDCF(Dcf_Prior,constants.CFN,constants.CFP,scores,labels)
    print(f"Min DCF for Polynomial Kernel SVM: {minDcf}\n")
    return minDcf

def K_Fold_SVM_kernel_rbf(D,L,K,hyperParameter_K,hyperParameter_C,hyperParameter_gamma,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=None,Calibration_Flag=None):
    # Leave-One-Out Approach Con K=2325: 
    fold_dimension = int(D.shape[1]/K)  # size of each fold
    fold_indices = numpy.arange(0, K*fold_dimension, fold_dimension)  # indices to split the data into folds
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
        if Z_Norm_Flag:
            # apply z-normalization
            DTR = normalization.zNormalizingData(DTR)
            DVA = normalization.zNormalizingData(DVA)
        if PCA_Flag and M!=None:
            # apply PCA on current fold DTR,DVA
            DTR,P = pca.PCA_projection(DTR,m = M)
            DVA = numpy.dot(P.T, DVA)
        nSamples = DVA.shape[1]  
        scores_i,nCorrectPrediction = svm.kernel_svm_radial(DTR, LTR, DVA, LVA, hyperParameter_K,hyperParameter_C,hyperParameter_gamma) 
        nWrongPrediction += nSamples - nCorrectPrediction
        scores = numpy.append(scores,scores_i)
        labels = numpy.append(labels,LVA)
    errorRate = nWrongPrediction/D.shape[1] 
    accuracy = 1 - errorRate
    print(f"RADIAL BASIS FUNCTION (RBF) Kernel SVM results:\nAccuracy: {round(accuracy*100, 2)}%\nError rate: {round(errorRate*100, 2)}%\n",end="")
    minDcf = optimal_decision.computeMinDCF(Dcf_Prior,constants.CFN,constants.CFP,scores,labels)
    print(f"Min DCF for RADIAL BASIS FUNCTION (RBF) Kernel SVM: {minDcf}\n")
    return minDcf

def K_Fold_GMM(D,L,K,nSplit0,nSplit1=None,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=None,Calibration_Flag=None):
    # Leave-One-Out Approach Con K=2325: 
    fold_dimension = int(D.shape[1]/K)  # size of each fold
    fold_indices = numpy.arange(0, K*fold_dimension, fold_dimension)  # indices to split the data into folds
    classifiers = [(gmm.LBGalgorithm,gmm.constraintSigma,"Full Covariance (standard)"), (gmm.DiagLBGalgorithm,gmm.DiagConstraintSigma,"Diagonal Covariance"), (gmm.TiedLBGalgorithm, gmm.constraintSigma, "Tied Covariance"),(gmm.TiedDiagLBGalgorithm,gmm.DiagConstraintSigma,"Tied Diagonal Covariance")] 
    #classifiers = [(gmm.TiedDiagLBGalgorithm,gmm.DiagConstraintSigma,"Tied Diagonal Covariance")] 
    # 4 values: mindcfs of Full Covariance, of Diagonal Covariance, of Tied Covariance, of Tied Diagonal Covariance
    minDcfs = []
    for classifier_algorithm, classifier_costraint, classifier_name in classifiers: 
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
            if Z_Norm_Flag:
                # apply z-normalization
                DTR = normalization.zNormalizingData(DTR)
                DVA = normalization.zNormalizingData(DVA)
            if PCA_Flag and M!=None:
                # apply PCA on current fold DTR,DVA
                DTR,P = pca.PCA_projection(DTR,m = M)
                DVA = numpy.dot(P.T, DVA)
            DTR0,DTR1 = main.getClassMatrix(DTR,LTR)
            nSamples = DVA.shape[1]
            scores_i,nCorrectPrediction = gmm.GMM_Classifier(DTR0, DTR1, DVA, LVA, classifier_algorithm, nSplit0 , nSplit1, classifier_costraint) 
            nWrongPrediction += nSamples - nCorrectPrediction
            scores = numpy.append(scores,scores_i)
            labels = numpy.append(labels,LVA)
        errorRate = nWrongPrediction/D.shape[1] 
        accuracy = 1 - errorRate
        print(f"{classifier_name} results:\nAccuracy: {round(accuracy*100, 2)}%\nError rate: {round(errorRate*100, 2)}%\n",end="")
        minDcf = optimal_decision.computeMinDCF(Dcf_Prior,constants.CFN,constants.CFP,scores,labels)
        minDcfs.append(minDcf)
        print(f"Min DCF for {classifier_name}: {minDcf}\n")
        #plot.compute_bayes_error_plot(scores,labels,"GMM")
    return minDcfs 

def optimalDecision(DTR,LTR,DTE,LTE):
    classifiers = [(generative_models.MVG_log_classifier, "Multivariate Gaussian Classifier"), (generative_models.NaiveBayesGaussianClassifier_log, "Naive Bayes"), (generative_models.TiedCovarianceGaussianClassifier_log, "Tied Covariance"), (generative_models.TiedNaiveBayesGaussianClassifier_log, "Tied Naive Bayes"),(lr.LogisticRegressionWeighted, "Logistic Regression Weighted"),(lr.LogisticRegressionWeightedQuadratic, "Logistic Regression Quadratic Weighted")] 
    for classifier_function, classifier_name in classifiers:
        LP,_ = classifier_function(DTR,LTR,DTE,LTE)
        # se sorto i log likelihoods mi dà problemi tutte le minDCF a 0
        LP = numpy.sort(LP)
        print(f"Min DCF for {classifier_name}: {optimal_decision.computeMinDCF(constants.PRIOR_PROBABILITY,constants.CFN,constants.CFP,LP,LTE)}")
