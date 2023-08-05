import numpy
import normalization
import pca
import lr
import svm
import gmm
import optimal_decision
import constants
import plot
import kfold
import main
import pickle
from plot_utility import PlotUtility

def best_model_score_calibration(DTR_RAND,LTR_RAND,DTE,LTE):
    m = 8
    # BEST MODEL QLR WITH lambda=10^-3, no PCA + ZNORM
    classifier = [(lr.LogisticRegressionWeightedQuadratic, "Logistic Regression Weighted Quadratic")]
    _,scores_train,labels_train = kfold.K_Fold_LR(DTR_RAND,LTR_RAND,K=constants.K,classifiers=classifier,lambd=0.001,PCA_Flag=None,M=None,Z_Norm_Flag=True,Dcf_Prior=0.5,Calibration_Flag=False)
    classifier = [(lr.LogisticRegressionWeightedQuadratic, "Logistic Regression Weighted Quadratic Evaluation")]
    _,scores_eval,labels_eval = LR_Eval(DTR_RAND,LTR_RAND,DTE,LTE,classifier,lambd=0.001,PCA_Flag=None,M=None,Z_Norm_Flag=True,Dcf_Prior=0.5)
    # ----- MISCALIBRATED PLOT --------
    plot.compute_bayes_error_plot(scores_eval,labels_eval,"Miscalibrated Logistic Regression Weighted Quadratic Evaluation")
    # ----- CALIBRATION AND CALIBRATED PLOT ------
    scores_train = numpy.hstack(scores_train).reshape(1,scores_train.shape[0])
    scores_eval = numpy.hstack(scores_eval).reshape(1,scores_eval.shape[0])
    LR_Calibration_Eval(scores_train,labels_train,scores_eval,labels_eval,plt_title="Calibrated Logistic Regression Weighted Quadratic Evaluation",Calibration_Flag=True)
    
    
    # BEST MODEL POLYNOMIAL SVM WITH c=1, C=10^-2, PCA + ZNORM
    _,scores_train,labels_train = kfold.K_Fold_SVM_kernel_polynomial(DTR_RAND,LTR_RAND,constants.K,hyperParameter_K=1,hyperParameter_C=0.01,hyperParameter_c=1,hyperParameter_d=2,PCA_Flag=True,M=m,Z_Norm_Flag=True,Dcf_Prior=0.5,Calibration_Flag=False)
    _,scores_eval,labels_eval = SVM_kernel_polynomial_Eval(DTR_RAND,LTR_RAND,DTE,LTE,hyperParameter_K=1,hyperParameter_C=0.01,hyperParameter_c=1,hyperParameter_d=2,PCA_Flag=True,M=m,Z_Norm_Flag=True,Dcf_Prior=0.5)
    # ----- MISCALIBRATED PLOT --------
    plot.compute_bayes_error_plot(scores_eval,labels_eval,"Miscalibrated Polynomial Kernel SVM Evaluation")
    # ----- CALIBRATION AND CALIBRATED PLOT ------
    scores_train = numpy.hstack(scores_train).reshape(1,scores_train.shape[0])
    scores_eval = numpy.hstack(scores_eval).reshape(1,scores_eval.shape[0])
    LR_Calibration_Eval(scores_train,labels_train,scores_eval,labels_eval,plt_title="Calibrated Polynomial Kernel SVM Evaluation",Calibration_Flag=True)

    # BEST MODEL RAW DIAGONAL WITH GMM COMPONENTS = 8 FOR CLASS 0 AND GMM COMPONENTS 2 FOR CLASS 1
    # HERE CALIBRATION NOT NEEDED, SO ONLY EVAL FOR FIRST BAYES ERROR PLOT
    classifiers = [(gmm.DiagLBGalgorithm,gmm.DiagConstraintSigma,"Diagonal Covariance Evaluation")]
    _,scores_eval,labels_eval = GMM_Eval(DTR_RAND,LTR_RAND,DTE,LTE,classifiers,nSplit0=3,nSplit1=1,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=0.5)
    # ----- ALREADY CALIBRATED PLOT --------
    plot.compute_bayes_error_plot(scores_eval,labels_eval,"Diagonal Covariance Evaluation")

def model_fusion(DTR_RAND,LTR_RAND,DTE,LTE):
    m = 8
    # BEST MODEL QLR WITH lambda=10^-3, no PCA + ZNORM
    classifier = [(lr.LogisticRegressionWeightedQuadratic, "Logistic Regression Weighted Quadratic")]
    _,scores_train_lr,labels_train = kfold.K_Fold_LR(DTR_RAND,LTR_RAND,K=constants.K,classifiers=classifier,lambd=0.001,PCA_Flag=None,M=None,Z_Norm_Flag=True,Dcf_Prior=0.5,Calibration_Flag=False)
    classifier = [(lr.LogisticRegressionWeightedQuadratic, "Logistic Regression Weighted Quadratic Evaluation")]
    _,scores_eval_lr,labels_eval = LR_Eval(DTR_RAND,LTR_RAND,DTE,LTE,classifier,lambd=0.001,PCA_Flag=None,M=None,Z_Norm_Flag=True,Dcf_Prior=0.5)
    # ---- TAKE CALIBRATION OUTPUT WITHOUT PLOTTING ----
    scores_tr = numpy.hstack(scores_train_lr).reshape(1,scores_train_lr.shape[0])
    scores_ev = numpy.hstack(scores_eval_lr).reshape(1,scores_eval_lr.shape[0])
    scores_lr = LR_Calibration_Eval(scores_tr,labels_train,scores_ev,labels_eval,plt_title=None,Calibration_Flag=None)

    # BEST MODEL POLYNOMIAL SVM WITH c=1, C=10^-2, PCA + ZNORM
    _,scores_pol_svm_train,labels_train = kfold.K_Fold_SVM_kernel_polynomial(DTR_RAND,LTR_RAND,constants.K,hyperParameter_K=1,hyperParameter_C=0.01,hyperParameter_c=1,hyperParameter_d=2,PCA_Flag=True,M=m,Z_Norm_Flag=True,Dcf_Prior=0.5,Calibration_Flag=False)
    _,scores_pol_svm_eval,labels_eval = SVM_kernel_polynomial_Eval(DTR_RAND,LTR_RAND,DTE,LTE,hyperParameter_K=1,hyperParameter_C=0.01,hyperParameter_c=1,hyperParameter_d=2,PCA_Flag=True,M=m,Z_Norm_Flag=True,Dcf_Prior=0.5)
    # ---- TAKE CALIBRATION OUTPUT WITHOUT PLOTTING ----
    scores_tr = numpy.hstack(scores_pol_svm_train).reshape(1,scores_pol_svm_train.shape[0])
    scores_ev = numpy.hstack(scores_pol_svm_eval).reshape(1,scores_pol_svm_eval.shape[0])
    scores_svm = LR_Calibration_Eval(scores_tr,labels_train,scores_ev,labels_eval,plt_title=None,Calibration_Flag=None)

    # BEST MODEL RAW DIAGONAL WITH GMM COMPONENTS = 8 FOR CLASS 0 AND GMM COMPONENTS 2 FOR CLASS 1
    classifiers = [(gmm.DiagLBGalgorithm,gmm.DiagConstraintSigma,"Diagonal Covariance")]
    _,scores_train_gmm,labels_train = kfold.K_Fold_GMM(DTR_RAND,LTR_RAND,K=constants.K,classifiers=classifiers,nSplit0=3,nSplit1=1,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=0.5,Calibration_Flag=False)
    classifiers = [(gmm.DiagLBGalgorithm,gmm.DiagConstraintSigma,"Diagonal Covariance Evaluation")]
    _,scores_eval_gmm,labels_eval = GMM_Eval(DTR_RAND,LTR_RAND,DTE,LTE,classifiers,nSplit0=3,nSplit1=1,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=0.5)
    # ---- TAKE CALIBRATION OUTPUT WITHOUT PLOTTING ----
    scores_tr = numpy.hstack(scores_train_gmm).reshape(1,scores_train_gmm.shape[0])
    scores_ev = numpy.hstack(scores_eval_gmm).reshape(1,scores_eval_gmm.shape[0])
    scores_gmm = LR_Calibration_Eval(scores_tr,labels_train,scores_ev,labels_eval,plt_title=None,Calibration_Flag=None)

    # QLR + SVM
    s_eval = [numpy.hstack(scores_lr),numpy.hstack(scores_svm)]
    s_train = [numpy.hstack(scores_train_lr),numpy.hstack(scores_pol_svm_train)]
    s_vstack_eval = numpy.vstack(s_eval)
    s_vstack_train = numpy.hstack(s_train)
    print("QLR + SVM")
    # ---- TAKE CALIBRATION OUTPUT WITHOUT PLOTTING, SINCE HSTACK IS NEEDED ----
    qlr_svm_eval_cal = LR_Calibration_Eval(s_vstack_train,LTR_RAND,s_vstack_eval,LTE,plt_title = None,Calibration_Flag=None)
    plot.compute_bayes_error_plot(numpy.hstack[qlr_svm_eval_cal],LTE,plt_title = "Model Fusion QLR + SVM Evaluation")

    # QLR + GMM
    s_eval = [numpy.hstack(scores_lr),numpy.hstack(scores_gmm)]
    s_train = [numpy.hstack(scores_train_lr),numpy.hstack(scores_train_gmm)]
    s_vstack_eval = numpy.vstack(s_eval)
    s_vstack_train = numpy.hstack(s_train)
    print("QLR + GMM")
    # ---- TAKE CALIBRATION OUTPUT WITHOUT PLOTTING, SINCE HSTACK IS NEEDED ----
    qlr_gmm_eval_cal = LR_Calibration_Eval(s_vstack_train,LTR_RAND,s_vstack_eval,LTE,plt_title = None,Calibration_Flag=None)
    plot.compute_bayes_error_plot(numpy.hstack[qlr_gmm_eval_cal],LTE,plt_title = "Model Fusion QLR + GMM Evaluation")
  
    # SVM + GMM
    s_eval = [numpy.hstack(scores_svm),numpy.hstack(scores_gmm)]
    s_train = [numpy.hstack(scores_pol_svm_train),numpy.hstack(scores_train_gmm)]
    s_vstack_eval = numpy.vstack(s_eval)
    s_vstack_train = numpy.hstack(s_train)
    print("SVM + GMM")
    # ---- TAKE CALIBRATION OUTPUT WITHOUT PLOTTING, SINCE HSTACK IS NEEDED ----
    svm_gmm_eval_cal = LR_Calibration_Eval(s_vstack_train,LTR_RAND,s_vstack_eval,LTE,plt_title = None,Calibration_Flag=None)
    plot.compute_bayes_error_plot(numpy.hstack[svm_gmm_eval_cal],LTE,plt_title = "Model Fusion SVM + GMM Evaluation")

    # QLR + SVM + GMM
    s_eval = [numpy.hstack(scores_lr),numpy.hstack(scores_svm),numpy.hstack(scores_gmm)]
    s_train = [numpy.hstack(scores_train_lr),numpy.hstack(scores_pol_svm_train),numpy.hstack(scores_train_gmm)]
    s_vstack_eval = numpy.vstack(s_eval)
    s_vstack_train = numpy.hstack(s_train)
    print("QLR + SVM + GMM")
    # ---- TAKE CALIBRATION OUTPUT WITHOUT PLOTTING, SINCE HSTACK IS NEEDED ----
    qlr_svm_gmm_eval_cal = LR_Calibration_Eval(s_vstack_train,LTR_RAND,s_vstack_eval,LTE,plt_title = None,Calibration_Flag=None)
    plot.compute_bayes_error_plot(numpy.hstack[qlr_svm_gmm_eval_cal],LTE,plt_title = "Model Fusion QLR + SVM + GMM Evaluation")

# CALIBRATION/FUSION LR
def LR_Calibration_Eval(DTR,LTR,DTE,LTE,plt_title,Calibration_Flag=None):
    nWrongPrediction = 0
    scores = numpy.array([])
    labels = numpy.array([])
    nSamples = DTE.shape[1]  
    scores_i,nCorrectPrediction = lr.LogisticRegressionPriorWeighted(DTR, LTR, DTE, LTE) 
    nWrongPrediction += nSamples - nCorrectPrediction
    scores = numpy.append(scores,scores_i)
    labels = numpy.append(labels,LTE)
    #errorRate = nWrongPrediction/D.shape[0] 
    #accuracy = 1 - errorRate
    #print(f"{classifier_name} results:\nAccuracy: {round(accuracy*100, 2)}%\nError rate: {round(errorRate*100, 2)}%\n",end="")
    #minDcf = optimal_decision.computeMinDCF(constants.PRIOR_PROBABILITY,constants.CFN,constants.CFP,scores,labels)
    #minDcfs.append(minDcf)
    #print(f"Min DCF for {classifier_name}: {minDcf}\n")
    if Calibration_Flag:
        plot.compute_bayes_error_plot(scores,labels,plt_title)
    return scores

# QLR
def LR_Eval(DTR,LTR,DTE,LTE,classifiers,lambd,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=None):
    minDcfs = []
    for classifier_function, classifier_name in classifiers: 
        nWrongPrediction = 0
        scores = numpy.array([])
        labels = numpy.array([])
      
        if Z_Norm_Flag:
            # apply z-normalization
            DTR = normalization.zNormalizingData(DTR)
            DTE = normalization.zNormalizingData(DTE)
        if PCA_Flag and M!=None:
            # apply PCA
            DTR,P = pca.PCA_projection(DTR,m = M)
            DTE = numpy.dot(P.T, DTE)
        nSamples = DTE.shape[1]  
        scores_i,nCorrectPrediction = classifier_function(DTR, LTR, DTE, LTE, lambd) 
        nWrongPrediction += nSamples - nCorrectPrediction
        scores = numpy.append(scores,scores_i)
        labels = numpy.append(labels,LTE)
        errorRate = nWrongPrediction/nSamples
        accuracy = 1 - errorRate
        print(f"{classifier_name} results:\nAccuracy: {round(accuracy*100, 2)}%\nError rate: {round(errorRate*100, 2)}%\n",end="")
        minDcf = optimal_decision.computeMinDCF(Dcf_Prior,constants.CFN,constants.CFP,scores,labels)
        minDcfs.append(minDcf)
        print(f"Min DCF for {classifier_name}: {minDcf}\n")
    # returns only scores and labels of the last passed classifier in classifiers
    return minDcfs,scores,labels

# POL_SVM
def SVM_kernel_polynomial_Eval(DTR,LTR,DTE,LTE,hyperParameter_K,hyperParameter_C,hyperParameter_c,hyperParameter_d,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=None):
    nWrongPrediction = 0
    scores = numpy.array([])
    labels = numpy.array([])
    
    if Z_Norm_Flag:
        # apply z-normalization
        DTR = normalization.zNormalizingData(DTR)
        DTE = normalization.zNormalizingData(DTE)
    if PCA_Flag and M!=None:
        # apply PCA on current fold DTR,DVA
        DTR,P = pca.PCA_projection(DTR,m = M)
        DTE = numpy.dot(P.T, DTE)
    nSamples = DTE.shape[1]  
    scores_i,nCorrectPrediction = svm.kernel_svm_polynomial(DTR, LTR, DTE, LTE, hyperParameter_K,hyperParameter_C,hyperParameter_c,hyperParameter_d) 
    nWrongPrediction += nSamples - nCorrectPrediction
    #print("Correct: " + str(nCorrectPrediction))
    #print("Wrong: " + str(nWrongPrediction))
    scores = numpy.append(scores,scores_i)
    labels = numpy.append(labels,LTE)
    errorRate = nWrongPrediction/nSamples
    accuracy = 1 - errorRate
    print(f"Polynomial Kernel SVM results:\nAccuracy: {round(accuracy*100, 2)}%\nError rate: {round(errorRate*100, 2)}%\n",end="")
    minDcf = optimal_decision.computeMinDCF(Dcf_Prior,constants.CFN,constants.CFP,scores,labels)
    print(f"Min DCF for Polynomial Kernel SVM: {minDcf}\n")
    return minDcf,scores,labels

# GMM
def GMM_Eval(DTR,LTR,DTE,LTE,classifiers,nSplit0,nSplit1=None,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=None):
    #classifiers = [(gmm.LBGalgorithm,gmm.constraintSigma,"Full Covariance (standard)"), (gmm.DiagLBGalgorithm,gmm.DiagConstraintSigma,"Diagonal Covariance"), (gmm.TiedLBGalgorithm, gmm.constraintSigma, "Tied Covariance"),(gmm.TiedDiagLBGalgorithm,gmm.DiagConstraintSigma,"Tied Diagonal Covariance")] 
    #classifiers = [(gmm.TiedDiagLBGalgorithm,gmm.DiagConstraintSigma,"Tied Diagonal Covariance")] 
    # 4 values: mindcfs of Full Covariance, of Diagonal Covariance, of Tied Covariance, of Tied Diagonal Covariance
    minDcfs = []
    for classifier_algorithm, classifier_costraint, classifier_name in classifiers: 
        nWrongPrediction = 0
        scores = numpy.array([])
        labels = numpy.array([])
        
        if Z_Norm_Flag:
            # apply z-normalization
            DTR = normalization.zNormalizingData(DTR)
            DTE = normalization.zNormalizingData(DTE)
        if PCA_Flag and M!=None:
            # apply PCA on current fold DTR,DVA
            DTR,P = pca.PCA_projection(DTR,m = M)
            DTE = numpy.dot(P.T, DTE)
        DTR0,DTR1 = main.getClassMatrix(DTR,LTR)
        nSamples = DTE.shape[1]
        scores_i,nCorrectPrediction = gmm.GMM_Classifier(DTR0, DTR1, DTE, LTE, classifier_algorithm, nSplit0 , nSplit1, classifier_costraint) 
        nWrongPrediction += nSamples - nCorrectPrediction
        scores = numpy.append(scores,scores_i)
        labels = numpy.append(labels,LTE)
        errorRate = nWrongPrediction/nSamples
        accuracy = 1 - errorRate
        print(f"{classifier_name} results:\nAccuracy: {round(accuracy*100, 2)}%\nError rate: {round(errorRate*100, 2)}%\n",end="")
        minDcf = optimal_decision.computeMinDCF(Dcf_Prior,constants.CFN,constants.CFP,scores,labels)
        minDcfs.append(minDcf)
        print(f"Min DCF for {classifier_name}: {minDcf}\n")
    return minDcfs,scores,labels 

# ---- PARAMETERS EVALUATION FOR FINDING EVENTUAL SUB-OPTIMAL SOLUTIONS ----

def eval_qlr_lambda_parameter_testing(DTR_RAND,LTR_RAND,DTE,LTE,Load_Data=False):
    prior = 0.5    
    lambda_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    zNorm_quadratic_eval = []
    classifier = [(lr.LogisticRegressionWeightedQuadratic, "Logistic Regression Weighted Quadratic Evaluation")]
    
    if not Load_Data:
        for lambd in lambda_values:
            print("lambda value : " + str(lambd))
            minDcfs,_,_ = LR_Eval(DTR_RAND,LTR_RAND,DTE,LTE,classifier,lambd=lambd,PCA_Flag=None,M=None,Z_Norm_Flag=True,Dcf_Prior=prior)
            zNorm_quadratic_eval.append(minDcfs[0])
        
        # Save the list of objects to a file
        with open("modelData/zNorm_quadratic_lr_eval" + str(prior) + ".pkl", "wb") as f:
            pickle.dump(zNorm_quadratic_eval, f)
    
    if Load_Data:
        # Retrieve the list of objects from the file
        with open("modelData/zNorm_quadratic_lr_eval" + str(prior) + ".pkl", "rb") as f:
            zNorm_quadratic_eval = pickle.load(f)
    
    # Retrieve the list of objects from the file
    with open("modelData/zNorm_quadratic_lr" + str(prior) + ".pkl", "rb") as f:
        zNorm_quadratic_train = pickle.load(f)
    
    # ------ PLOT LR_QUADRATIC ------
    labels = ['minDCF [eval set]','minDCF [valid set]']
    colors = ['b','g']
    # array of lambda values (for linear) and corresponding mindcfs
    plot.plotDCF([lambda_values,lambda_values],[zNorm_quadratic_eval,zNorm_quadratic_train],labels,colors,'lambda',title='Quadratic Logistic Regression Evaluation')

def eval_svm_kernel_polynomial_C_c_parameter_testing(DTR_RAND,LTR_RAND,DTE,LTE,Load_Data=False):
    # BEST MODEL POLYNOMIAL SVM WITH c=1, C=10^-2, PCA + ZNORM
    prior = 0.5
    m = 8
    C_values = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0] # for C <= 10^-6 there is a significative worsening in performance 
    c_values = [0, 1]
    zNormPca_polynomial_eval = []

    if not Load_Data:
        for C in C_values:
            print("C value : " + str(C))
            for c in c_values:
                print("c value : " + str(c))
                minDcf,_,_ = SVM_kernel_polynomial_Eval(DTR_RAND,LTR_RAND,DTE,LTE,hyperParameter_K=1,hyperParameter_C=C,hyperParameter_c=c,hyperParameter_d=2,PCA_Flag=True,M=m,Z_Norm_Flag=True,Dcf_Prior=prior)
                zNormPca_polynomial_eval.append(PlotUtility(prior=prior,k=1,C=C,c=c,d=2,minDcf=minDcf))
        
        # Save the list of objects to a file
        with open("modelData/zNormPca_polynomial_svm_eval" + str(prior) + ".pkl", "wb") as f:
            pickle.dump(zNormPca_polynomial_eval, f)
    
    if Load_Data:
        # Retrieve the list of objects from the file
        with open("modelData/zNormPca_polynomial_svm_eval" + str(prior) + ".pkl", "rb") as f:
            zNormPca_polynomial_eval = pickle.load(f)
    
    # Retrieve the list of objects from the file
    with open("modelData/zNormPca_polynomial_svm" + str(prior) + ".pkl", "rb") as f:
        zNormPca_polynomial_train = pickle.load(f)
    
    # ----- PLOT FOR c = 0 ------
    zNormPca_polynomial_c0_train = list(filter(lambda PlotElement: PlotElement.is_c(0), zNormPca_polynomial_train))
    minDcfs_zNormPca_polynomial_c0_train = [PlotElement.getminDcf() for PlotElement in zNormPca_polynomial_c0_train]
    C_values_zNormPca_polynomial_c0_train = [PlotElement.getC() for PlotElement in zNormPca_polynomial_c0_train]

    zNormPca_polynomial_c0_eval = list(filter(lambda PlotElement: PlotElement.is_c(0), zNormPca_polynomial_eval))
    minDcfs_zNormPca_polynomial_c0_eval = [PlotElement.getminDcf() for PlotElement in zNormPca_polynomial_c0_eval]
    C_values_zNormPca_polynomial_c0_eval = [PlotElement.getC() for PlotElement in zNormPca_polynomial_c0_eval]

    labels = ['minDCF [eval set] c = 0','minDCF [valid set] c = 0']
    colors = ['b','g']
    plot.plotDCF([C_values_zNormPca_polynomial_c0_eval,C_values_zNormPca_polynomial_c0_train],[minDcfs_zNormPca_polynomial_c0_eval,minDcfs_zNormPca_polynomial_c0_train],labels,colors,xlabel='C',title='Polynomial SVM Evaluation')


    # ----- PLOT FOR c = 1 ------
    zNormPca_polynomial_c1_train = list(filter(lambda PlotElement: PlotElement.is_c(1), zNormPca_polynomial_train))
    minDcfs_zNormPca_polynomial_c1_train = [PlotElement.getminDcf() for PlotElement in zNormPca_polynomial_c1_train]
    C_values_zNormPca_polynomial_c1_train = [PlotElement.getC() for PlotElement in zNormPca_polynomial_c1_train]

    zNormPca_polynomial_c1_eval = list(filter(lambda PlotElement: PlotElement.is_c(1), zNormPca_polynomial_eval))
    minDcfs_zNormPca_polynomial_c1_eval = [PlotElement.getminDcf() for PlotElement in zNormPca_polynomial_c1_eval]
    C_values_zNormPca_polynomial_c1_eval = [PlotElement.getC() for PlotElement in zNormPca_polynomial_c1_eval]

    labels = ['minDCF [eval set] c = 1','minDCF [valid set] c = 1']
    colors = ['b','g']
    plot.plotDCF([C_values_zNormPca_polynomial_c1_eval,C_values_zNormPca_polynomial_c1_train],[minDcfs_zNormPca_polynomial_c1_eval,minDcfs_zNormPca_polynomial_c1_train],labels,colors,xlabel='C',title='Polynomial SVM Evaluation')

def eval_GMMAllRawCombinations(DTR_RAND,LTR_RAND,DTE,LTE,Load_Data=False):
    # ---------- GMM WITH ALL POSSIBLE RAW COMPONENTS COMBINATION -----------
    classifiers = [(gmm.LBGalgorithm,gmm.constraintSigma,"Full Covariance (standard) Evaluation"), (gmm.DiagLBGalgorithm,gmm.DiagConstraintSigma,"Diagonal Covariance Evaluation"), (gmm.TiedLBGalgorithm, gmm.constraintSigma, "Tied Covariance Evaluation"),(gmm.TiedDiagLBGalgorithm,gmm.DiagConstraintSigma,"Tied Diagonal Covariance Evaluation")]
    prior = 0.5
    if not Load_Data:
        colors = {
            0 : 'blue',
            1 : 'red',
            2 : 'green',
            3 : 'yellow',
            4 : 'magenta',
            5 : 'cyan',
            6 : 'black',
            7 : 'white'
        }
        print("GMM WITH ALL POSSIBLE COMPONENTS COMBINATION")
        labels_eval = []
        plot_colors_eval = []
        gmm_components_class_1_eval = []
        # mindcfs of Full Covariance, of Diagonal Covariance, of Tied Covariance, of Tied Diagonal Covariance
        raw_full_min_dcfs_eval = []
        raw_diag_min_dcfs_eval = []
        raw_tied_min_dcfs_eval = []
        raw_tied_diag_min_dcfs_eval = []
        for nSplit0 in range(0,4):
            print("Number of GMM Components of Class 0: " + str(2**nSplit0))
            labels_eval.append("minDCF G0 = " + str(2**nSplit0))
            plot_colors_eval.append(colors[nSplit0])
            gmm_components_class_1_single_eval = []

            raw_full_min_dcfs_single_eval = []
            raw_diag_min_dcfs_single_eval = []
            raw_tied_min_dcfs_single_eval = []
            raw_tied_diag_min_dcfs_single_eval = []
            for nSplit1 in range(0,4):
                print("Number of GMM Components of Class 1: " + str(2**nSplit1))
                gmm_components_class_1_single_eval.append(2**nSplit1)
                # minDcfs[0] mindcfs of Full Covariance, minDcfs[1] of Diagonal Covariance, minDcfs[2] of Tied Covariance, minDcfs[3] of Tied Diagonal Covariance

                raw_minDcfs,_,_ = GMM_Eval(DTR_RAND,LTR_RAND,DTE,LTE,classifiers,nSplit0=nSplit0,nSplit1=nSplit1,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=prior)
                        
                raw_full_min_dcfs_single_eval.append(raw_minDcfs[0])
                raw_diag_min_dcfs_single_eval.append(raw_minDcfs[1])
                raw_tied_min_dcfs_single_eval.append(raw_minDcfs[2])
                raw_tied_diag_min_dcfs_single_eval.append(raw_minDcfs[3])

            gmm_components_class_1_eval.append(gmm_components_class_1_single_eval)
                    
            raw_full_min_dcfs_eval.append(raw_full_min_dcfs_single_eval)
            raw_diag_min_dcfs_eval.append(raw_diag_min_dcfs_single_eval)
            raw_tied_min_dcfs_eval.append(raw_tied_min_dcfs_single_eval)
            raw_tied_diag_min_dcfs_eval.append(raw_tied_diag_min_dcfs_single_eval)

            # Save the list of objects to a file
            with open("modelData/labels_gmm_allComb_eval" + str(prior) + ".pkl", "wb") as f:
                pickle.dump(labels_eval, f)
                
            # Save the list of objects to a file
            with open("modelData/colors_gmm_allComb_eval" + str(prior) + ".pkl", "wb") as f:
                pickle.dump(plot_colors_eval, f)
                
            # Save the list of objects to a file
            with open("modelData/components_gmm_allComb_eval" + str(prior) + ".pkl", "wb") as f:
                pickle.dump(gmm_components_class_1_eval, f)

            # Save the list of objects to a file
            with open("modelData/raw_full_gmm_allComb_eval" + str(prior) + ".pkl", "wb") as f:
                pickle.dump(raw_full_min_dcfs_eval, f)
            # Save the list of objects to a file
            with open("modelData/raw_diag_gmm_allComb_eval" + str(prior) + ".pkl", "wb") as f:
                pickle.dump(raw_diag_min_dcfs_eval, f)
            # Save the list of objects to a file
            with open("modelData/raw_tied_gmm_allComb_eval" + str(prior) + ".pkl", "wb") as f:
                pickle.dump(raw_tied_min_dcfs_eval, f)
            # Save the list of objects to a file
            with open("modelData/raw_tied_diag_gmm_allComb_eval" + str(prior) + ".pkl", "wb") as f:
                pickle.dump(raw_tied_diag_min_dcfs_eval, f)
    
    if Load_Data:
        # Retrieve data for plotting

        # Retrieve the list of objects from the file
        with open("modelData/labels_gmm_allComb_eval" + str(prior) + ".pkl", "rb") as f:
            labels_eval = pickle.load(f)
            
        # Retrieve the list of objects from the file
        with open("modelData/colors_gmm_allComb_eval" + str(prior) + ".pkl", "rb") as f:
            plot_colors_eval = pickle.load(f)
            
        # Retrieve the list of objects from the file
        with open("modelData/components_gmm_allComb_eval" + str(prior) + ".pkl", "rb") as f:
            gmm_components_class_1_eval = pickle.load(f)
            
        # Retrieve the list of objects from the file
        with open("modelData/raw_full_gmm_allComb_eval" + str(prior) + ".pkl", "rb") as f:
            raw_full_min_dcfs_eval = pickle.load(f)
        # Retrieve the list of objects from the file
        with open("modelData/raw_diag_gmm_allComb_eval" + str(prior) + ".pkl", "rb") as f:
            raw_diag_min_dcfs_eval = pickle.load(f)
        # Retrieve the list of objects from the file
        with open("modelData/raw_tied_gmm_allComb_eval" + str(prior) + ".pkl", "rb") as f:
            raw_tied_min_dcfs_eval = pickle.load(f)
        # Retrieve the list of objects from the file
        with open("modelData/raw_tied_diag_gmm_allComb_eval" + str(prior) + ".pkl", "rb") as f:
            raw_tied_diag_min_dcfs_eval = pickle.load(f)
    
    # ----- PLOT GMMS ALL COMBINATIONS  ------

    # ----- RAW -------
    plot.gmm_plot_all_component_combinations(gmm_components_class_1_eval,raw_full_min_dcfs_eval,labels_eval,plot_colors_eval,"Full Covariance (standard) no PCA no Znorm Evaluation")
    plot.gmm_plot_all_component_combinations(gmm_components_class_1_eval,raw_diag_min_dcfs_eval,labels_eval,plot_colors_eval,"Diagonal Covariance no PCA no Znorm Evaluation")
    plot.gmm_plot_all_component_combinations(gmm_components_class_1_eval,raw_tied_min_dcfs_eval,labels_eval,plot_colors_eval,"Tied Covariance no PCA no Znorm Evaluation")
    plot.gmm_plot_all_component_combinations(gmm_components_class_1_eval,raw_tied_diag_min_dcfs_eval,labels_eval,plot_colors_eval,"Tied Diagonal Covariance no PCA no Znorm Evaluation")