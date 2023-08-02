import kfold
import lr
import constants
import gmm
import numpy

def best_model_score_calibration(DTR_RAND,LTR_RAND):
    m = 8
    # BEST MODEL QLR WITH lambda=10^-3, no PCA + ZNORM
    classifier = [(lr.LogisticRegressionWeightedQuadratic, "Logistic Regression Weighted Quadratic")]
    kfold.K_Fold_LR(DTR_RAND,LTR_RAND,K=constants.K,classifiers=classifier,lambd=0.001,PCA_Flag=None,M=None,Z_Norm_Flag=True,Dcf_Prior=0.5,Calibration_Flag=True)

    # BEST MODEL POLYNOMIAL SVM WITH c=1, C=10^-2, PCA + ZNORM
    kfold.K_Fold_SVM_kernel_polynomial(DTR_RAND,LTR_RAND,constants.K,hyperParameter_K=1,hyperParameter_C=0.01,hyperParameter_c=1,hyperParameter_d=2,PCA_Flag=True,M=m,Z_Norm_Flag=True,Dcf_Prior=0.5,Calibration_Flag=True)
    
    # BEST MODEL RAW DIAGONAL WITH GMM COMPONENTS = 8 FOR CLASS 0 AND GMM COMPONENTS 2 FOR CLASS 1
    classifiers = [(gmm.DiagLBGalgorithm,gmm.DiagConstraintSigma,"Diagonal Covariance")]
    kfold.K_Fold_GMM(DTR_RAND,LTR_RAND,K=constants.K,classifiers=classifiers,nSplit0=3,nSplit1=1,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=0.5,Calibration_Flag=True)

def model_fusion(DTR_RAND,LTR_RAND):
    m = 8

    classifier = [(lr.LogisticRegressionWeightedQuadratic, "Logistic Regression Weighted Quadratic")]
    _,scores_lr,labels_lr = kfold.K_Fold_LR(DTR_RAND,LTR_RAND,K=constants.K,classifiers=classifier,lambd=0.001,PCA_Flag=None,M=None,Z_Norm_Flag=True,Dcf_Prior=0.5,Calibration_Flag=False)

    classifiers = [(gmm.DiagLBGalgorithm,gmm.DiagConstraintSigma,"Diagonal Covariance")]
    _,scores_gmm,labels_gmm = kfold.K_Fold_GMM(DTR_RAND,LTR_RAND,K=constants.K,classifiers=classifiers,nSplit0=3,nSplit1=1,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=0.5,Calibration_Flag=False)

    _,scores_pol_svm,labels_pol_svm = kfold.K_Fold_SVM_kernel_polynomial(DTR_RAND,LTR_RAND,constants.K,hyperParameter_K=1,hyperParameter_C=0.01,hyperParameter_c=1,hyperParameter_d=2,PCA_Flag=True,M=m,Z_Norm_Flag=True,Dcf_Prior=0.5,Calibration_Flag=False)
    
    # QLR + SVM
    s = [numpy.hstack(scores_lr),numpy.hstack(scores_pol_svm)]
    s_new=numpy.vstack(s)
    print("QLR + SVM")
    kfold.K_Fold_Calibration(s_new,labels_lr,K=constants.K,plt_title="Model Fusion QLR + SVM",model_fusion=True,num_models=2)
    
    # QLR + GMM
    s = [numpy.hstack(scores_lr),numpy.hstack(scores_gmm)]
    s_new=numpy.vstack(s)
    print("QLR + GMM")
    kfold.K_Fold_Calibration(s_new,labels_lr,K=constants.K,plt_title="Model Fusion QLR + GMM",model_fusion=True,num_models=2)
    
    # SVM + GMM
    s = [numpy.hstack(scores_pol_svm),numpy.hstack(scores_gmm)]
    s_new=numpy.vstack(s)
    print("SVM + GMM")
    kfold.K_Fold_Calibration(s_new,labels_lr,K=constants.K,plt_title="Model Fusion SVM + GMM",model_fusion=True,num_models=2)

    # QLR + SVM + GMM
    s = [numpy.hstack(scores_lr),numpy.hstack(scores_pol_svm),numpy.hstack(scores_gmm)]
    s_new=numpy.vstack(s)
    print("QLR + SVM + GMM")
    kfold.K_Fold_Calibration(s_new,labels_lr,K=constants.K,plt_title="Model Fusion QLR + SVM + GMM",model_fusion=True,num_models=3)
