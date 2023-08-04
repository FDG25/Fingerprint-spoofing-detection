import numpy
import matplotlib.pyplot as plt
import constants
import lr
from plot_utility import PlotUtility
import parameter_tuning
import training
import kfold
import calibration_fusion
import gmm
import evaluation

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

    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # --------------- MODEL TRAINING ------------------------------
    # -------------------------------------------------------------
    # -------------------------------------------------------------

    # ---------------   GENERATIVE MODELS   -----------------------

    # training.trainGenerative(DTR_RAND,LTR_RAND)

    # ---------------   LR MODELS   -----------------------

    # training.trainLR(DTR_RAND,LTR_RAND,Load_Data=True)
    
    #print("No Weight")
    #lr.LogisticRegressionWeighted(DTR,LTR,DTE,LTE)
    #print("Weight")
    #lr.LogisticRegression(DTR,LTR,DTE,LTE)

    # ---------------   SVM MODELS   -----------------------

    # training.trainLinearSVM(DTR_RAND,LTR_RAND,Load_Data=True)
    # training.trainPolynomialSVM(DTR_RAND,LTR_RAND,Load_Data=True)
    # training.trainRadialBasisFunctionSVM(DTR_RAND,LTR_RAND,Load_Data=True)
    
    # -------------- GMM --------------------
    
    # training.trainGMMSameComponents(DTR_RAND,LTR_RAND,Load_Data=False)
    # training.trainGMMAllCombinations(DTR_RAND,LTR_RAND,Load_Data=True)
    
    # -------------- SCORE CALIBRATION -------------
    
    # calibration_fusion.best_model_score_calibration(DTR_RAND,LTR_RAND)

    # -------------- MODEL FUSION ----------------
    # calibration_fusion.model_fusion(DTR_RAND,LTR_RAND)

    # -------------------------------------------------------------
    # -------------------------------------------------------------
    # --------------- MODEL EVALUATION ----------------------------
    # -------------------------------------------------------------
    # -------------------------------------------------------------

    # -------------- SCORE CALIBRATION -------------

    # evaluation.best_model_score_calibration(DTR_RAND,LTR_RAND,DTE,LTE)

    # -------------- MODEL FUSION ----------------

    # evaluation.model_fusion(DTR_RAND,LTR_RAND,DTE,LTE)

    # ---------------   QLR MODEL   -----------------------

    # evaluation.eval_qlr_lambda_parameter_testing(DTR_RAND,LTR_RAND,DTE,LTE,Load_Data=False)

    # ---------------   POLYNOMIAL SVM MODEL   -----------------------

    # evaluation.eval_svm_kernel_polynomial_C_c_parameter_testing(DTR_RAND,LTR_RAND,DTE,LTE,Load_Data=False)

    # -------------- GMM --------------------

    # evaluation.eval_GMMAllRawCombinations(DTR_RAND,LTR_RAND,DTE,LTE,Load_Data=False)