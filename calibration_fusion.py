import kfold
import lr
import constants
import gmm
import numpy

def best_model_score_calibration(DTR_RAND,LTR_RAND):
    # ONCE DECIDED, REPLACE THE VALUES WITH THE BEST HYPERPARAMETER AND PREPROCESSING TECHNIQUE AND COMPUTE SCORE CALIBRATION
    classifier = [(lr.LogisticRegressionWeightedQuadratic, "Logistic Regression Weighted Quadratic")]
    kfold.K_Fold_LR(DTR_RAND,LTR_RAND,K=constants.K,classifiers=classifier,lambd=0.001,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=0.5,Calibration_Flag=True)

    kfold.K_Fold_SVM_kernel_polynomial(DTR_RAND,LTR_RAND,constants.K,hyperParameter_K=1,hyperParameter_C=0.1,hyperParameter_c=1,hyperParameter_d=2,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=0.5,Calibration_Flag=True)
    
    classifiers = [(gmm.TiedDiagLBGalgorithm,gmm.DiagConstraintSigma,"Tied Diagonal Covariance")]
    kfold.K_Fold_GMM(DTR_RAND,LTR_RAND,K=constants.K,classifiers=classifiers,nSplit0=3,nSplit1=1,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=0.5,Calibration_Flag=True)

def model_fusion(DTR_RAND,LTR_RAND):
    classifier = [(lr.LogisticRegressionWeightedQuadratic, "Logistic Regression Weighted Quadratic")]
    _,scores_lr,labels_lr = kfold.K_Fold_LR(DTR_RAND,LTR_RAND,K=constants.K,classifiers=classifier,lambd=0.001,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=0.5,Calibration_Flag=None)

    classifiers = [(gmm.TiedDiagLBGalgorithm,gmm.DiagConstraintSigma,"Tied Diagonal Covariance")]
    _,scores_gmm,labels_gmm = kfold.K_Fold_GMM(DTR_RAND,LTR_RAND,K=constants.K,classifiers=classifiers,nSplit0=3,nSplit1=1,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=0.5,Calibration_Flag=None)

    # PUT SCORES TOGETHER AND RESHUFFLE THEM
    s = [numpy.hstack(scores_lr),numpy.hstack(scores_gmm)]
    s_new=numpy.vstack(s)

    kfold.K_Fold_Calibration(s_new,labels_lr,K=constants.K,plt_title="Model Fusion QLR + GMM",model_fusion=True,num_models=2)
