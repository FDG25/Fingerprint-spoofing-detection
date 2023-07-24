import kfold
import parameter_tuning
import lr
import constants
import plot

def trainGenerative(DTR_RAND,LTR_RAND):
    # ---------------   GENERATIVE MODELS   -----------------------

    print("RAW (No PCA No Z_Norm)")
    print("Prior = 0.5")
    kfold.K_Fold_Generative(DTR_RAND,LTR_RAND,K=constants.K,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=0.5)
    print("Prior = 0.1")
    kfold.K_Fold_Generative(DTR_RAND,LTR_RAND,K=constants.K,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=0.1)
    print("Prior = 0.9")
    kfold.K_Fold_Generative(DTR_RAND,LTR_RAND,K=constants.K,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=0.9)
    
    # print("\nZ_Norm\n")
    # print("Prior = 0.5")
    # kfold.K_Fold_Generative(DTR_RAND,LTR_RAND,K=constants.K,PCA_Flag=None,M=None,Z_Norm_Flag=True,Dcf_Prior=0.5)
    # print("Prior = 0.1")
    # kfold.K_Fold_Generative(DTR_RAND,LTR_RAND,K=constants.K,PCA_Flag=None,M=None,Z_Norm_Flag=True,Dcf_Prior=0.1)
    # print("Prior = 0.9")
    # kfold.K_Fold_Generative(DTR_RAND,LTR_RAND,K=constants.K,PCA_Flag=None,M=None,Z_Norm_Flag=True,Dcf_Prior=0.9)

    for m in range(10,4,-1):
        print("RAW + PCA with M = " + str(m) + "\n")
        print("Prior = 0.5")
        kfold.K_Fold_Generative(DTR_RAND,LTR_RAND,K=constants.K,PCA_Flag=True,M=m,Z_Norm_Flag=None,Dcf_Prior=0.5)
        print("Prior = 0.1")
        kfold.K_Fold_Generative(DTR_RAND,LTR_RAND,K=constants.K,PCA_Flag=True,M=m,Z_Norm_Flag=None,Dcf_Prior=0.1)
        print("Prior = 0.9")
        kfold.K_Fold_Generative(DTR_RAND,LTR_RAND,K=constants.K,PCA_Flag=True,M=m,Z_Norm_Flag=None,Dcf_Prior=0.9)

        # print("\nZ_Norm + PCA with M = " + str(m) + "\n")
        # print("Prior = 0.5")
        # kfold.K_Fold_Generative(DTR_RAND,LTR_RAND,K=constants.K,PCA_Flag=True,M=m,Z_Norm_Flag=True,Dcf_Prior=0.5)
        # print("Prior = 0.1")
        # kfold.K_Fold_Generative(DTR_RAND,LTR_RAND,K=constants.K,PCA_Flag=True,M=m,Z_Norm_Flag=True,Dcf_Prior=0.1)
        # print("Prior = 0.9")
        # kfold.K_Fold_Generative(DTR_RAND,LTR_RAND,K=constants.K,PCA_Flag=True,M=m,Z_Norm_Flag=True,Dcf_Prior=0.9)

def trainLR(DTR_RAND,LTR_RAND):
    lambda_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    # classifier = [(lr.LogisticRegressionWeighted, "Logistic Regression Weighted"),(lr.LogisticRegressionWeightedQuadratic, "Logistic Regression Weighted Quadratic")]
    classifier = [(lr.LogisticRegressionWeighted, "Logistic Regression Weighted")]
    
    # ------ LR_LINEAR 0.5 ------

    print("Prior = 0.5\n")
    print("RAW (No PCA No Z_Norm)\n")
    raw_linear,_ = parameter_tuning.lr_lambda_parameter_testing(DTR_RAND,LTR_RAND,K=constants.K,lambda_values=lambda_values,classifier=classifier,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=0.5,Calibration_Flag=None)
    print("Z_Norm\n")
    zNorm_linear,_ = parameter_tuning.lr_lambda_parameter_testing(DTR_RAND,LTR_RAND,K=constants.K,lambda_values=lambda_values,classifier=classifier,PCA_Flag=None,M=None,Z_Norm_Flag=True,Dcf_Prior=0.5,Calibration_Flag=None)
    m = 8
    print("RAW + PCA with M = " + str(m) + "\n")
    rawPca_linear,_ = parameter_tuning.lr_lambda_parameter_testing(DTR_RAND,LTR_RAND,K=constants.K,lambda_values=lambda_values,classifier=classifier,PCA_Flag=True,M=m,Z_Norm_Flag=None,Dcf_Prior=0.5,Calibration_Flag=None)
    print("Z_Norm + PCA with M = " + str(m) + "\n")
    zNormPca_linear,_ = parameter_tuning.lr_lambda_parameter_testing(DTR_RAND,LTR_RAND,K=constants.K,lambda_values=lambda_values,classifier=classifier,PCA_Flag=True,M=m,Z_Norm_Flag=True,Dcf_Prior=0.5,Calibration_Flag=None)

    # ------ PLOT LR_LINEAR 0.5 ------
    labels = ['minDCF no PCA no Znorm','minDCF no PCA Znorm','minDCF PCA=' + str(m) + ' no Znorm','minDCF PCA=' + str(m) + ' Znorm']
    colors = ['b','r','g','y']
    # array of lambda values (for linear) and corresponding mindcfs
    plot.plotDCF([lambda_values,lambda_values,lambda_values,lambda_values],[raw_linear,zNorm_linear,rawPca_linear,zNormPca_linear],labels,colors,xlabel='lambda',title='Linear Logistic Regression with $\pi=0.5$')

    # ------ LR_LINEAR 0.1 ------

    print("Prior = 0.1\n")
    print("RAW (No PCA No Z_Norm)\n")
    raw_linear,_ = parameter_tuning.lr_lambda_parameter_testing(DTR_RAND,LTR_RAND,K=constants.K,lambda_values=lambda_values,classifier=classifier,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=0.1,Calibration_Flag=None)
    print("Z_Norm\n")
    zNorm_linear,_ = parameter_tuning.lr_lambda_parameter_testing(DTR_RAND,LTR_RAND,K=constants.K,lambda_values=lambda_values,classifier=classifier,PCA_Flag=None,M=None,Z_Norm_Flag=True,Dcf_Prior=0.1,Calibration_Flag=None)
    m = 8
    print("RAW + PCA with M = " + str(m) + "\n")
    rawPca_linear,_ = parameter_tuning.lr_lambda_parameter_testing(DTR_RAND,LTR_RAND,K=constants.K,lambda_values=lambda_values,classifier=classifier,PCA_Flag=True,M=m,Z_Norm_Flag=None,Dcf_Prior=0.1,Calibration_Flag=None)
    print("Z_Norm + PCA with M = " + str(m) + "\n")
    zNormPca_linear,_ = parameter_tuning.lr_lambda_parameter_testing(DTR_RAND,LTR_RAND,K=constants.K,lambda_values=lambda_values,classifier=classifier,PCA_Flag=True,M=m,Z_Norm_Flag=True,Dcf_Prior=0.1,Calibration_Flag=None)

    # ------ PLOT LR_LINEAR 0.1 ------
    labels = ['minDCF no PCA no Znorm','minDCF no PCA Znorm','minDCF PCA=' + str(m) + ' no Znorm','minDCF PCA=' + str(m) + ' Znorm']
    colors = ['b','r','g','y']
    # array of lambda values (for linear) and corresponding mindcfs
    plot.plotDCF([lambda_values,lambda_values,lambda_values,lambda_values],[raw_linear,zNorm_linear,rawPca_linear,zNormPca_linear],labels,colors,xlabel='lambda',title='Linear Logistic Regression with $\pi=0.1$')
    
    # ------ LR_LINEAR 0.9 ------

    print("Prior = 0.9\n")
    print("RAW (No PCA No Z_Norm)\n")
    raw_linear,_ = parameter_tuning.lr_lambda_parameter_testing(DTR_RAND,LTR_RAND,K=constants.K,lambda_values=lambda_values,classifier=classifier,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=0.9,Calibration_Flag=None)
    print("Z_Norm\n")
    zNorm_linear,_ = parameter_tuning.lr_lambda_parameter_testing(DTR_RAND,LTR_RAND,K=constants.K,lambda_values=lambda_values,classifier=classifier,PCA_Flag=None,M=None,Z_Norm_Flag=True,Dcf_Prior=0.9,Calibration_Flag=None)
    m = 8
    print("RAW + PCA with M = " + str(m) + "\n")
    rawPca_linear,_ = parameter_tuning.lr_lambda_parameter_testing(DTR_RAND,LTR_RAND,K=constants.K,lambda_values=lambda_values,classifier=classifier,PCA_Flag=True,M=m,Z_Norm_Flag=None,Dcf_Prior=0.9,Calibration_Flag=None)
    print("Z_Norm + PCA with M = " + str(m) + "\n")
    zNormPca_linear,_ = parameter_tuning.lr_lambda_parameter_testing(DTR_RAND,LTR_RAND,K=constants.K,lambda_values=lambda_values,classifier=classifier,PCA_Flag=True,M=m,Z_Norm_Flag=True,Dcf_Prior=0.9,Calibration_Flag=None)

    # ------ PLOT LR_LINEAR 0.9 ------
    labels = ['minDCF no PCA no Znorm','minDCF no PCA Znorm','minDCF PCA=' + str(m) + ' no Znorm','minDCF PCA=' + str(m) + ' Znorm']
    colors = ['b','r','g','y']
    # array of lambda values (for linear) and corresponding mindcfs
    plot.plotDCF([lambda_values,lambda_values,lambda_values,lambda_values],[raw_linear,zNorm_linear,rawPca_linear,zNormPca_linear],labels,colors,xlabel='lambda',title='Linear Logistic Regression with $\pi=0.9$')
    
    # # ------ PLOT LR_QUADRATIC ------
    # labels = ['Quadratic']
    # colors = ['b']
    # # array of lambda values (for linear) and corresponding mindcfs
    # plot.plotDCF([lambda_values],[minDcfs_Quadratic],labels,colors,'lambda')