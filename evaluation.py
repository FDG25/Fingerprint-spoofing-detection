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
    _,scores_eval,labels_eval = GMM_Eval(DTR_RAND,LTR_RAND,DTE,LTE,classifiers,nSplit0=3,nSplit1=1,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=0.5,Calibration_Flag=True)
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

# PLO_SVM
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

