import kfold
import lr
import constants
import gmm

def best_model_score_calibration(DTR_RAND,LTR_RAND):
    # ONCE DECIDED, REPLACE THE VALUES WITH THE BEST HYPERPARAMETER AND PREPROCESSING TECHNIQUE AND COMPUTE SCORE CALIBRATION
    classifier = [(lr.LogisticRegressionWeightedQuadratic, "Logistic Regression Weighted Quadratic")]
    kfold.K_Fold_LR(DTR_RAND,LTR_RAND,K=constants.K,classifiers=classifier,lambd=0.001,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=0.5,Calibration_Flag=True)

    kfold.K_Fold_SVM_kernel_polynomial(DTR_RAND,LTR_RAND,constants.K,hyperParameter_K=1,hyperParameter_C=0.1,hyperParameter_c=1,hyperParameter_d=2,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=0.5,Calibration_Flag=True)
    
    classifiers = [(gmm.TiedDiagLBGalgorithm,gmm.DiagConstraintSigma,"Tied Diagonal Covariance")]
    kfold.K_Fold_GMM(DTR_RAND,LTR_RAND,K=constants.K,classifiers=classifiers,nSplit0=3,nSplit1=1,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=0.5,Calibration_Flag=True)