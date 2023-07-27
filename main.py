import numpy
import matplotlib.pyplot as plt
import constants
import lr
from plot_utility import PlotUtility
import parameter_tuning
import training
import kfold

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
    #plot.plot_fraction_explained_variance_pca(DTR)
    #plot.plot_Heatmap_Whole_Dataset(DTR)
    #plot.plot_Heatmap_Spoofed_Authentic(DTR,LTR,Class_Label=0)
    #plot.plot_Heatmap_Spoofed_Authentic(DTR,LTR,Class_Label=1)
    # RANDOMIZE DATASET BEFORE K-FOLD
    DTR_RAND,LTR_RAND = randomize(DTR,LTR)
    DTE_RAND,LTE_RAND = randomize(DTE,LTE)

    # ---------------   GENERATIVE MODELS   -----------------------

    # training.trainGenerative(DTR_RAND,LTR_RAND)

    # ---------------   LR MODELS   -----------------------

    # training.trainLR(DTR_RAND,LTR_RAND,Load_Data=False)
    #print("No Weight")
    #lr.LogisticRegressionWeighted(DTR,LTR,DTE,LTE)
    #print("Weight")
    #lr.LogisticRegression(DTR,LTR,DTE,LTE)

    # ---------------   SVM MODELS   -----------------------

    # training.trainLinearSVM(DTR_RAND,LTR_RAND,Load_Data=False)
    # training.trainPolynomialSVM(DTR_RAND,LTR_RAND,Load_Data=False)
    # training.trainRadialBasisFunctionSVM(DTR_RAND,LTR_RAND,Load_Data=False)
    
    # -------------- GMM --------------------
    
    # training.trainGMMSameComponents(DTR_RAND,LTR_RAND,Load_Data=False)
    training.trainGMMAllCombinations(DTR_RAND,LTR_RAND,Load_Data=True)
    
    # BEST MODEL TIED DIAGONAL WITH GMM COMPONENTS = 8 FOR CLASS 0 AND GMM COMPONENTS 2 FOR CLASS 1
    # minDcfs = kfold.K_Fold_GMM(DTR_RAND,LTR_RAND,K=5,nSplit0=3,nSplit1=1)

    classifier = [(lr.LogisticRegressionWeightedQuadratic, "Logistic Regression Weighted Quadratic")]
    minDcfs = kfold.K_Fold_LR(DTR_RAND,LTR_RAND,K=5,classifiers=classifier,hyperParameter=0.01)
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

