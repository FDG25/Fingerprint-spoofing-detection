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

def K_Fold_SVM(D,L,K,classifiers,hyperParameter_K,hyperParameter_C):
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
            scores_i,nCorrectPrediction = classifier_function(DTR, LTR, DVA, LVA, hyperParameter_K,hyperParameter_C) 
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
    
def svm_K_C_parameters_testing(DTR,LTR,k_values,c_values,classifier):  
    priors = [constants.PRIOR_PROBABILITY]
    minDcfs=[]
    for i in range(len(priors)):
        print("prior:",priors[i])        
        for k_value in k_values:
            print("k value : " + str(k_value))
            for c in c_values:
                print("c value : " + str(c))
                minDcf = K_Fold_SVM(DTR,LTR,K=5,classifiers=classifier,hyperParameter_K=k_value,hyperParameter_C=c)
                minDcfs.append(minDcf)
            # PLOT for each K value
            print("plot for k value : " + str(k_value))
            plot.plotDCF(c_values,minDcfs,'lambda')

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
    #print("K_Fold with K = 5")
    #print("PCA with m = " + str(constants.M))
    #lambda_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
    #classifier = [(lr.LogisticRegressionWeighted, "Logistic Regression Weighted"),(lr.LogisticRegressionWeightedQuadratic, "Logistic Regression Weighted Quadratic")]
    #lr_lambda_parameter_testing(DTR_RAND,LTR_RAND,lambda_values,'lambda',classifier)
    #print("Leave One Out (K = 2325)")
    #K_Fold(DTR_RAND,LTR_RAND,K=2325)
    #print("No Weight")
    #lr.LogisticRegressionWeighted(DTR,LTR,DTE,LTE)
    #print("Weight")
    #lr.LogisticRegression(DTR,LTR,DTE,LTE)


    # ---------------   SVM MODELS   -----------------------
    print("SVM HYPERPARAMETERS K AND C TESTING:")
    K_values = [1, 10]
    C_values = [0.1, 1.0, 10.0, 100.0]
    classifier = [(svm.linear_svm, "Linear SVM")]
    svm_K_C_parameters_testing(DTR_RAND,LTR_RAND,K_values,C_values,classifier)
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

