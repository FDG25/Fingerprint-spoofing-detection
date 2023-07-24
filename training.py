import kfold
import parameter_tuning
import lr
import constants
import plot
import parameter_tuning
import numpy

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
    # ---------------   LR MODELS   -----------------------
    lambda_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    # classifier = [(lr.LogisticRegressionWeighted, "Logistic Regression Weighted"),(lr.LogisticRegressionWeightedQuadratic, "Logistic Regression Weighted Quadratic")]
    classifier = [(lr.LogisticRegressionWeighted, "Logistic Regression Weighted")]
    
    for prior in [0.5,0.1,0.9]:
        # ------ LR_LINEAR ------
        print("Prior = " + str(prior) + "\n")
        print("RAW (No PCA No Z_Norm)\n")
        raw_linear,_ = parameter_tuning.lr_lambda_parameter_testing(DTR_RAND,LTR_RAND,K=constants.K,lambda_values=lambda_values,classifier=classifier,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=prior,Calibration_Flag=None)
        print("Z_Norm\n")
        zNorm_linear,_ = parameter_tuning.lr_lambda_parameter_testing(DTR_RAND,LTR_RAND,K=constants.K,lambda_values=lambda_values,classifier=classifier,PCA_Flag=None,M=None,Z_Norm_Flag=True,Dcf_Prior=prior,Calibration_Flag=None)
        m = 8
        print("RAW + PCA with M = " + str(m) + "\n")
        rawPca_linear,_ = parameter_tuning.lr_lambda_parameter_testing(DTR_RAND,LTR_RAND,K=constants.K,lambda_values=lambda_values,classifier=classifier,PCA_Flag=True,M=m,Z_Norm_Flag=None,Dcf_Prior=prior,Calibration_Flag=None)
        print("Z_Norm + PCA with M = " + str(m) + "\n")
        zNormPca_linear,_ = parameter_tuning.lr_lambda_parameter_testing(DTR_RAND,LTR_RAND,K=constants.K,lambda_values=lambda_values,classifier=classifier,PCA_Flag=True,M=m,Z_Norm_Flag=True,Dcf_Prior=prior,Calibration_Flag=None)

        # ------ PLOT LR_LINEAR ------
        labels = ['minDCF no PCA no Znorm','minDCF no PCA Znorm','minDCF PCA=' + str(m) + ' no Znorm','minDCF PCA=' + str(m) + ' Znorm']
        colors = ['b','r','g','y']
        # array of lambda values (for linear) and corresponding mindcfs
        plot.plotDCF([lambda_values,lambda_values,lambda_values,lambda_values],[raw_linear,zNorm_linear,rawPca_linear,zNormPca_linear],labels,colors,xlabel='lambda',title='Linear Logistic Regression with $\pi=' + str(prior) + '$')

        # # ------ PLOT LR_QUADRATIC ------
        # labels = ['Quadratic']
        # colors = ['b']
        # # array of lambda values (for linear) and corresponding mindcfs
        # plot.plotDCF([lambda_values],[minDcfs_Quadratic],labels,colors,'lambda')

def trainLinearSVM(DTR_RAND,LTR_RAND):
    # ---------------   LINEAR SVM   -----------------------
    print("SVM LINEAR HYPERPARAMETERS K AND C TESTING:")
    K_values = [1, 10] # K=10 migliore ma tutor ha detto valore lab 1 vedere, al max 1,10 
    C_values = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    for prior in [0.5,0.1,0.9]:
        print("Prior = " + str(prior) + "\n")

        print("RAW (No PCA No Z_Norm)\n")
        raw_linear = parameter_tuning.svm_linear_K_C_parameters_testing(DTR_RAND,LTR_RAND,K=constants.K,k_values=K_values,C_values=C_values,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=prior,Calibration_Flag=None)
        
        print("Z_Norm\n")
        zNorm_linear = parameter_tuning.svm_linear_K_C_parameters_testing(DTR_RAND,LTR_RAND,K=constants.K,k_values=K_values,C_values=C_values,PCA_Flag=None,M=None,Z_Norm_Flag=True,Dcf_Prior=prior,Calibration_Flag=None)
        
        m = 8
        print("RAW + PCA with M = " + str(m) + "\n")
        pca_linear = parameter_tuning.svm_linear_K_C_parameters_testing(DTR_RAND,LTR_RAND,K=constants.K,k_values=K_values,C_values=C_values,PCA_Flag=True,M=m,Z_Norm_Flag=None,Dcf_Prior=prior,Calibration_Flag=None)
        
        print("Z_Norm + PCA with M = " + str(m) + "\n")
        zNormPca_linear = parameter_tuning.svm_linear_K_C_parameters_testing(DTR_RAND,LTR_RAND,K=constants.K,k_values=K_values,C_values=C_values,PCA_Flag=True,M=m,Z_Norm_Flag=True,Dcf_Prior=prior,Calibration_Flag=None)

        # ----  SINGLE PLOT FOR prior, K = 1  -----
        raw_linear_k_1 = list(filter(lambda PlotElement: PlotElement.is_k(1), raw_linear))
        minDcfs_raw_linear_k_1 = [PlotElement.getminDcf() for PlotElement in raw_linear_k_1]
        C_values_raw_linear_k_1 = [PlotElement.getC() for PlotElement in raw_linear_k_1]

        zNorm_linear_k_1 = list(filter(lambda PlotElement: PlotElement.is_k(1), zNorm_linear))
        minDcfs_zNorm_linear_k_1 = [PlotElement.getminDcf() for PlotElement in zNorm_linear_k_1]
        C_values_zNorm_linear_k_1 = [PlotElement.getC() for PlotElement in zNorm_linear_k_1]

        pca_linear_k_1 = list(filter(lambda PlotElement: PlotElement.is_k(1), pca_linear))
        minDcfs_pca_linear_k_1 = [PlotElement.getminDcf() for PlotElement in pca_linear_k_1]
        C_values_pca_linear_k_1 = [PlotElement.getC() for PlotElement in pca_linear_k_1]

        zNormPca_linear_k_1 = list(filter(lambda PlotElement: PlotElement.is_k(1), zNormPca_linear))
        minDcfs_zNormPca_linear_k_1 = [PlotElement.getminDcf() for PlotElement in zNormPca_linear_k_1]
        C_values_zNormPca_linear_k_1 = [PlotElement.getC() for PlotElement in zNormPca_linear_k_1]

        labels = ['minDCF K = 1 no PCA no Znorm','minDCF K = 1 no PCA Znorm','minDCF K = 1 PCA=' + str(m) + ' no Znorm','minDCF K = 1 PCA=' + str(m) + ' Znorm']
        colors = ['b','r','g','y']
        #base colors: r, g, b, m, y, c, k, w
        plot.plotDCF([C_values_raw_linear_k_1,C_values_zNorm_linear_k_1,C_values_pca_linear_k_1,C_values_zNormPca_linear_k_1],[minDcfs_raw_linear_k_1,minDcfs_zNorm_linear_k_1,minDcfs_pca_linear_k_1,minDcfs_zNormPca_linear_k_1],labels,colors,xlabel='C',title='Linear SVM with $\pi=' + str(prior) + '$')

        # ----  SINGLE PLOT FOR Prior = 0.5, K = 10  -----
        raw_linear_k_10 = list(filter(lambda PlotElement: PlotElement.is_k(10), raw_linear))
        minDcfs_raw_linear_k_10 = [PlotElement.getminDcf() for PlotElement in raw_linear_k_10]
        C_values_raw_linear_k_10 = [PlotElement.getC() for PlotElement in raw_linear_k_10]

        zNorm_linear_k_10 = list(filter(lambda PlotElement: PlotElement.is_k(10), zNorm_linear))
        minDcfs_zNorm_linear_k_10 = [PlotElement.getminDcf() for PlotElement in zNorm_linear_k_10]
        C_values_zNorm_linear_k_10 = [PlotElement.getC() for PlotElement in zNorm_linear_k_10]

        pca_linear_k_10 = list(filter(lambda PlotElement: PlotElement.is_k(10), pca_linear))
        minDcfs_pca_linear_k_10 = [PlotElement.getminDcf() for PlotElement in pca_linear_k_10]
        C_values_pca_linear_k_10 = [PlotElement.getC() for PlotElement in pca_linear_k_10]

        zNormPca_linear_k_10 = list(filter(lambda PlotElement: PlotElement.is_k(10), zNormPca_linear))
        minDcfs_zNormPca_linear_k_10 = [PlotElement.getminDcf() for PlotElement in zNormPca_linear_k_10]
        C_values_zNormPca_linear_k_10 = [PlotElement.getC() for PlotElement in zNormPca_linear_k_10]

        labels = ['minDCF K = 10 no PCA no Znorm','minDCF K = 10 no PCA Znorm','minDCF K = 10 PCA=' + str(m) + ' no Znorm','minDCF K = 10 PCA=' + str(m) + ' Znorm']
        colors = ['b','r','g','y']
        #base colors: r, g, b, m, y, c, k, w
        plot.plotDCF([C_values_raw_linear_k_10,C_values_zNorm_linear_k_10,C_values_pca_linear_k_10,C_values_zNormPca_linear_k_10],[minDcfs_raw_linear_k_10,minDcfs_zNorm_linear_k_10,minDcfs_pca_linear_k_10,minDcfs_zNormPca_linear_k_10],labels,colors,xlabel='C',title='Linear SVM with $\pi=' + str(prior) + '$')


def trainPolynomialSVM(DTR_RAND,LTR_RAND):
    pass
    # print("SVM POLYNOMIAL K,C,c,d TESTING:")
    # K_values = [1]
    # C_values = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0] # for C <= 10^-6 there is a significative worsening in performance 
    # c_values = [0, 1]
    # d_values = [2.0]
    # parameter_tuning.svm_kernel_polynomial_K_C_c_d_parameter_testing(DTR_RAND,LTR_RAND,K_values,C_values,c_values,d_values)

def trainRadialBasisFunctionSVM(DTR_RAND,LTR_RAND):
    pass
    # print("SVM RADIAL BASIS FUNCTION (RBF) K,C,gamma TESTING:")
    # K_values = [1.0]
    # C_values = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    # # we want log(gamma), so we pass gamma value for which log(gamma) = -1,-2,-3,-4,-5
    # gamma_values = [1.0/numpy.exp(3), 1.0/numpy.exp(4), 1.0/numpy.exp(5)] #hyper-parameter
    # parameter_tuning.svm_kernel_rbf_K_C_gamma_parameter_testing(DTR_RAND,LTR_RAND,K_values,C_values,gamma_values)