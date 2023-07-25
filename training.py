import kfold
import parameter_tuning
import lr
import constants
import plot
import parameter_tuning
import numpy
import pickle

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

def trainLR(DTR_RAND,LTR_RAND,Load_Data=False):
    # ---------------   LR MODELS   -----------------------
    lambda_values = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    # classifier = [(lr.LogisticRegressionWeighted, "Logistic Regression Weighted"),(lr.LogisticRegressionWeightedQuadratic, "Logistic Regression Weighted Quadratic")]
    classifier = [(lr.LogisticRegressionWeighted, "Logistic Regression Weighted")]
    
    for prior in constants.DCFS_PRIORS:
        # ------ LR_LINEAR ------
        if not Load_Data:
            print("Prior = " + str(prior) + "\n")
            print("RAW (No PCA No Z_Norm)\n")
            raw_linear,_ = parameter_tuning.lr_lambda_parameter_testing(DTR_RAND,LTR_RAND,K=constants.K,lambda_values=lambda_values,classifier=classifier,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=prior,Calibration_Flag=None)
            # Save the list of objects to a file
            with open("modelData/raw_linear_lr" + str(prior) + ".pkl", "wb") as f:
                pickle.dump(raw_linear, f)
            print("Z_Norm\n")
            zNorm_linear,_ = parameter_tuning.lr_lambda_parameter_testing(DTR_RAND,LTR_RAND,K=constants.K,lambda_values=lambda_values,classifier=classifier,PCA_Flag=None,M=None,Z_Norm_Flag=True,Dcf_Prior=prior,Calibration_Flag=None)
            # Save the list of objects to a file
            with open("modelData/zNorm_linear_lr" + str(prior) + ".pkl", "wb") as f:
                pickle.dump(zNorm_linear, f)
            m = 8
            print("RAW + PCA with M = " + str(m) + "\n")
            rawPca_linear,_ = parameter_tuning.lr_lambda_parameter_testing(DTR_RAND,LTR_RAND,K=constants.K,lambda_values=lambda_values,classifier=classifier,PCA_Flag=True,M=m,Z_Norm_Flag=None,Dcf_Prior=prior,Calibration_Flag=None)
            # Save the list of objects to a file
            with open("modelData/rawPca_linear_lr" + str(prior) + ".pkl", "wb") as f:
                pickle.dump(rawPca_linear, f)
            print("Z_Norm + PCA with M = " + str(m) + "\n")
            zNormPca_linear,_ = parameter_tuning.lr_lambda_parameter_testing(DTR_RAND,LTR_RAND,K=constants.K,lambda_values=lambda_values,classifier=classifier,PCA_Flag=True,M=m,Z_Norm_Flag=True,Dcf_Prior=prior,Calibration_Flag=None)
            # Save the list of objects to a file
            with open("modelData/zNormPca_linear_lr" + str(prior) + ".pkl", "wb") as f:
                pickle.dump(zNormPca_linear, f)

        if Load_Data:
            # Retrieve the list of objects from the file
            with open("modelData/raw_linear_lr" + str(prior) + ".pkl", "rb") as f:
                raw_linear = pickle.load(f)
            # Retrieve the list of objects from the file
            with open("modelData/zNorm_linear_lr" + str(prior) + ".pkl", "rb") as f:
                zNorm_linear = pickle.load(f)
            # Retrieve the list of objects from the file
            with open("modelData/rawPca_linear_lr" + str(prior) + ".pkl", "rb") as f:
                rawPca_linear = pickle.load(f)
            # Retrieve the list of objects from the file
            with open("modelData/zNormPca_linear_lr" + str(prior) + ".pkl", "rb") as f:
                zNormPca_linear = pickle.load(f)
        m = 8
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
    print("SVM LINEAR HYPERPARAMETERS K AND C TRAINING:")
    K_values = [1, 10] # K=10 migliore ma tutor ha detto valore lab 1 vedere, al max 1,10 
    C_values = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    for prior in constants.DCFS_PRIORS:
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

        # ----  SINGLE PLOT FOR prior, K = 10  -----
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
    print("SVM POLYNOMIAL K,C,c,d TRAINING:")
    K_values = [1]
    C_values = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0] # for C <= 10^-6 there is a significative worsening in performance 
    c_values = [0, 1]
    d_values = [2.0]

    for prior in constants.DCFS_PRIORS:
        print("Prior = " + str(prior) + "\n")

        print("RAW (No PCA No Z_Norm)\n")
        raw_polynomial = parameter_tuning.svm_kernel_polynomial_K_C_c_d_parameter_testing(DTR_RAND,LTR_RAND,K=constants.K,k_values=K_values,C_values=C_values,c_values=c_values,d_values=d_values,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=prior,Calibration_Flag=None)
        print("Z_Norm\n")
        zNorm_polynomial = parameter_tuning.svm_kernel_polynomial_K_C_c_d_parameter_testing(DTR_RAND,LTR_RAND,K=constants.K,k_values=K_values,C_values=C_values,c_values=c_values,d_values=d_values,PCA_Flag=None,M=None,Z_Norm_Flag=True,Dcf_Prior=prior,Calibration_Flag=None)
        m = 8
        print("RAW + PCA with M = " + str(m) + "\n")
        pca_polynomial = parameter_tuning.svm_kernel_polynomial_K_C_c_d_parameter_testing(DTR_RAND,LTR_RAND,K=constants.K,k_values=K_values,C_values=C_values,c_values=c_values,d_values=d_values,PCA_Flag=True,M=m,Z_Norm_Flag=None,Dcf_Prior=prior,Calibration_Flag=None)
        print("Z_Norm + PCA with M = " + str(m) + "\n")
        zNormPca_polynomial = parameter_tuning.svm_kernel_polynomial_K_C_c_d_parameter_testing(DTR_RAND,LTR_RAND,K=constants.K,k_values=K_values,C_values=C_values,c_values=c_values,d_values=d_values,PCA_Flag=True,M=m,Z_Norm_Flag=True,Dcf_Prior=prior,Calibration_Flag=None)
        
        
        # ----- PLOT FOR c = 0 ------
        raw_polynomial_c0 = list(filter(lambda PlotElement: PlotElement.is_c(0), raw_polynomial))
        minDcfs_raw_polynomial_c0 = [PlotElement.getminDcf() for PlotElement in raw_polynomial_c0]
        C_values_raw_polynomial_c0 = [PlotElement.getC() for PlotElement in raw_polynomial_c0]

        zNorm_polynomial_c0 = list(filter(lambda PlotElement: PlotElement.is_c(0), zNorm_polynomial))
        minDcfs_zNorm_polynomial_c0 = [PlotElement.getminDcf() for PlotElement in zNorm_polynomial_c0]
        C_values_zNorm_polynomial_c0 = [PlotElement.getC() for PlotElement in zNorm_polynomial_c0]

        pca_polynomial_c0 = list(filter(lambda PlotElement: PlotElement.is_c(0), pca_polynomial))
        minDcfs_pca_polynomial_c0 = [PlotElement.getminDcf() for PlotElement in pca_polynomial_c0]
        C_values_pca_polynomial_c0 = [PlotElement.getC() for PlotElement in pca_polynomial_c0]

        zNormPca_polynomial_c0 = list(filter(lambda PlotElement: PlotElement.is_c(0), zNormPca_polynomial))
        minDcfs_zNormPca_polynomial_c0 = [PlotElement.getminDcf() for PlotElement in zNormPca_polynomial_c0]
        C_values_zNormPca_polynomial_c0 = [PlotElement.getC() for PlotElement in zNormPca_polynomial_c0]

        labels = ['minDCF c = 0 no PCA no Znorm','minDCF c = 0 no PCA Znorm','minDCF c = 0 PCA=' + str(m) + ' no Znorm','minDCF c = 0 PCA=' + str(m) + ' Znorm']
        colors = ['b','r','g','y']
        #base colors: r, g, b, m, y, c, k, w
        plot.plotDCF([C_values_raw_polynomial_c0,C_values_zNorm_polynomial_c0,C_values_pca_polynomial_c0,C_values_zNormPca_polynomial_c0],[minDcfs_raw_polynomial_c0,minDcfs_zNorm_polynomial_c0,minDcfs_pca_polynomial_c0,minDcfs_zNormPca_polynomial_c0],labels,colors,xlabel='C',title='Polynomial SVM with $\pi=' + str(prior) + '$')


        # ----- PLOT FOR c = 1 ------
        raw_polynomial_c1 = list(filter(lambda PlotElement: PlotElement.is_c(1), raw_polynomial))
        minDcfs_raw_polynomial_c1 = [PlotElement.getminDcf() for PlotElement in raw_polynomial_c1]
        C_values_raw_polynomial_c1 = [PlotElement.getC() for PlotElement in raw_polynomial_c1]

        zNorm_polynomial_c1 = list(filter(lambda PlotElement: PlotElement.is_c(1), zNorm_polynomial))
        minDcfs_zNorm_polynomial_c1 = [PlotElement.getminDcf() for PlotElement in zNorm_polynomial_c1]
        C_values_zNorm_polynomial_c1 = [PlotElement.getC() for PlotElement in zNorm_polynomial_c1]

        pca_polynomial_c1 = list(filter(lambda PlotElement: PlotElement.is_c(1), pca_polynomial))
        minDcfs_pca_polynomial_c1 = [PlotElement.getminDcf() for PlotElement in pca_polynomial_c1]
        C_values_pca_polynomial_c1 = [PlotElement.getC() for PlotElement in pca_polynomial_c1]

        zNormPca_polynomial_c1 = list(filter(lambda PlotElement: PlotElement.is_c(1), zNormPca_polynomial))
        minDcfs_zNormPca_polynomial_c1 = [PlotElement.getminDcf() for PlotElement in zNormPca_polynomial_c1]
        C_values_zNormPca_polynomial_c1 = [PlotElement.getC() for PlotElement in zNormPca_polynomial_c1]

        labels = ['minDCF c = 1 no PCA no Znorm','minDCF c = 1 no PCA Znorm','minDCF c = 1 PCA=' + str(m) + ' no Znorm','minDCF c = 1 PCA=' + str(m) + ' Znorm']
        colors = ['b','r','g','y']
        #base colors: r, g, b, m, y, c, k, w
        plot.plotDCF([C_values_raw_polynomial_c1,C_values_zNorm_polynomial_c1,C_values_pca_polynomial_c1,C_values_zNormPca_polynomial_c1],[minDcfs_raw_polynomial_c1,minDcfs_zNorm_polynomial_c1,minDcfs_pca_polynomial_c1,minDcfs_zNormPca_polynomial_c1],labels,colors,xlabel='C',title='Polynomial SVM with $\pi=' + str(prior) + '$')

def trainRadialBasisFunctionSVM(DTR_RAND,LTR_RAND):
    print("SVM RADIAL BASIS FUNCTION (RBF) K,C,gamma TRAINING:")
    K_values = [1.0]
    C_values = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    # we want log(gamma), so we pass gamma value for which log(gamma) = -3,-4,-5
    gamma_values = [1.0/numpy.exp(3), 1.0/numpy.exp(4), 1.0/numpy.exp(5)] #hyper-parameter
    
    for prior in constants.DCFS_PRIORS:
        print("Prior = " + str(prior) + "\n")

        print("RAW (No PCA No Z_Norm)\n")
        raw_rbf = parameter_tuning.svm_kernel_rbf_K_C_gamma_parameter_testing(DTR_RAND,LTR_RAND,K=constants.K,k_values=K_values,C_values=C_values,gamma_values=gamma_values,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=prior,Calibration_Flag=None)
        print("Z_Norm\n")
        zNorm_rbf = parameter_tuning.svm_kernel_rbf_K_C_gamma_parameter_testing(DTR_RAND,LTR_RAND,K=constants.K,k_values=K_values,C_values=C_values,gamma_values=gamma_values,PCA_Flag=None,M=None,Z_Norm_Flag=True,Dcf_Prior=prior,Calibration_Flag=None)
        m = 8
        print("RAW + PCA with M = " + str(m) + "\n")
        pca_rbf = parameter_tuning.svm_kernel_rbf_K_C_gamma_parameter_testing(DTR_RAND,LTR_RAND,K=constants.K,k_values=K_values,C_values=C_values,gamma_values=gamma_values,PCA_Flag=True,M=m,Z_Norm_Flag=None,Dcf_Prior=prior,Calibration_Flag=None)
        print("Z_Norm + PCA with M = " + str(m) + "\n")
        zNormPca_rbf = parameter_tuning.svm_kernel_rbf_K_C_gamma_parameter_testing(DTR_RAND,LTR_RAND,K=constants.K,k_values=K_values,C_values=C_values,gamma_values=gamma_values,PCA_Flag=True,M=m,Z_Norm_Flag=True,Dcf_Prior=prior,Calibration_Flag=None)
        
        # ---- PLOT FOR log(gamma) = -3 -----
        raw_gamma_1e3 = list(filter(lambda PlotElement: PlotElement.is_gamma(1/numpy.exp(3)), raw_rbf))
        minDcfs_raw_gamma_1e3 = [PlotElement.getminDcf() for PlotElement in raw_gamma_1e3]
        C_values_raw_gamma_1e3 = [PlotElement.getC() for PlotElement in raw_gamma_1e3]

        zNorm_gamma_1e3 = list(filter(lambda PlotElement: PlotElement.is_gamma(1/numpy.exp(3)), zNorm_rbf))
        minDcfs_zNorm_gamma_1e3 = [PlotElement.getminDcf() for PlotElement in zNorm_gamma_1e3]
        C_values_zNorm_gamma_1e3 = [PlotElement.getC() for PlotElement in zNorm_gamma_1e3]

        pca_gamma_1e3 = list(filter(lambda PlotElement: PlotElement.is_gamma(1/numpy.exp(3)), pca_rbf))
        minDcfs_pca_gamma_1e3 = [PlotElement.getminDcf() for PlotElement in pca_gamma_1e3]
        C_values_pca_gamma_1e3 = [PlotElement.getC() for PlotElement in pca_gamma_1e3]

        zNormPca_gamma_1e3 = list(filter(lambda PlotElement: PlotElement.is_gamma(1/numpy.exp(3)), zNormPca_rbf))
        minDcfs_zNormPca_gamma_1e3 = [PlotElement.getminDcf() for PlotElement in zNormPca_gamma_1e3]
        C_values_zNormPca_gamma_1e3 = [PlotElement.getC() for PlotElement in zNormPca_gamma_1e3]

        labels = ['minDCF log(γ) = -3 no PCA no Znorm','minDCF log(γ) = -3 no PCA Znorm','minDCF log(γ) = -3 PCA=' + str(m) + ' no Znorm','minDCF log(γ) = -3 PCA=' + str(m) + ' Znorm']
        colors = ['b','g','y','c','r']
        #base colors: r, g, b, m, y, c, k, w
        plot.plotDCF([C_values_raw_gamma_1e3,C_values_zNorm_gamma_1e3,C_values_pca_gamma_1e3,C_values_zNormPca_gamma_1e3],[minDcfs_raw_gamma_1e3,minDcfs_zNorm_gamma_1e3,minDcfs_pca_gamma_1e3,minDcfs_zNormPca_gamma_1e3],labels,colors,xlabel='C',title='RBF SVM with $\pi=' + str(prior) + '$')


        # ---- PLOT FOR log(gamma) = -4 -----
        raw_gamma_1e4 = list(filter(lambda PlotElement: PlotElement.is_gamma(1/numpy.exp(4)), raw_rbf))
        minDcfs_raw_gamma_1e4 = [PlotElement.getminDcf() for PlotElement in raw_gamma_1e4]
        C_values_raw_gamma_1e4 = [PlotElement.getC() for PlotElement in raw_gamma_1e4]

        zNorm_gamma_1e4 = list(filter(lambda PlotElement: PlotElement.is_gamma(1/numpy.exp(4)), zNorm_rbf))
        minDcfs_zNorm_gamma_1e4 = [PlotElement.getminDcf() for PlotElement in zNorm_gamma_1e4]
        C_values_zNorm_gamma_1e4 = [PlotElement.getC() for PlotElement in zNorm_gamma_1e4]

        pca_gamma_1e4 = list(filter(lambda PlotElement: PlotElement.is_gamma(1/numpy.exp(4)), pca_rbf))
        minDcfs_pca_gamma_1e4 = [PlotElement.getminDcf() for PlotElement in pca_gamma_1e4]
        C_values_pca_gamma_1e4 = [PlotElement.getC() for PlotElement in pca_gamma_1e4]

        zNormPca_gamma_1e4 = list(filter(lambda PlotElement: PlotElement.is_gamma(1/numpy.exp(4)), zNormPca_rbf))
        minDcfs_zNormPca_gamma_1e4 = [PlotElement.getminDcf() for PlotElement in zNormPca_gamma_1e4]
        C_values_zNormPca_gamma_1e4 = [PlotElement.getC() for PlotElement in zNormPca_gamma_1e4]

        labels = ['minDCF log(γ) = -4 no PCA no Znorm','minDCF log(γ) = -4 no PCA Znorm','minDCF log(γ) = -4 PCA=' + str(m) + ' no Znorm','minDCF log(γ) = -4 PCA=' + str(m) + ' Znorm']
        colors = ['b','g','y','c','r']
        #base colors: r, g, b, m, y, c, k, w
        plot.plotDCF([C_values_raw_gamma_1e4,C_values_zNorm_gamma_1e4,C_values_pca_gamma_1e4,C_values_zNormPca_gamma_1e4],[minDcfs_raw_gamma_1e4,minDcfs_zNorm_gamma_1e4,minDcfs_pca_gamma_1e4,minDcfs_zNormPca_gamma_1e4],labels,colors,xlabel='C',title='RBF SVM with $\pi=' + str(prior) + '$')


        # ---- PLOT FOR log(gamma) = -5 -----
        raw_gamma_1e5 = list(filter(lambda PlotElement: PlotElement.is_gamma(1/numpy.exp(5)), raw_rbf))
        minDcfs_raw_gamma_1e5 = [PlotElement.getminDcf() for PlotElement in raw_gamma_1e5]
        C_values_raw_gamma_1e5 = [PlotElement.getC() for PlotElement in raw_gamma_1e5]

        zNorm_gamma_1e5 = list(filter(lambda PlotElement: PlotElement.is_gamma(1/numpy.exp(5)), zNorm_rbf))
        minDcfs_zNorm_gamma_1e5 = [PlotElement.getminDcf() for PlotElement in zNorm_gamma_1e5]
        C_values_zNorm_gamma_1e5 = [PlotElement.getC() for PlotElement in zNorm_gamma_1e5]

        pca_gamma_1e5 = list(filter(lambda PlotElement: PlotElement.is_gamma(1/numpy.exp(5)), pca_rbf))
        minDcfs_pca_gamma_1e5 = [PlotElement.getminDcf() for PlotElement in pca_gamma_1e5]
        C_values_pca_gamma_1e5 = [PlotElement.getC() for PlotElement in pca_gamma_1e5]

        zNormPca_gamma_1e5 = list(filter(lambda PlotElement: PlotElement.is_gamma(1/numpy.exp(5)), zNormPca_rbf))
        minDcfs_zNormPca_gamma_1e5 = [PlotElement.getminDcf() for PlotElement in zNormPca_gamma_1e5]
        C_values_zNormPca_gamma_1e5 = [PlotElement.getC() for PlotElement in zNormPca_gamma_1e5]

        labels = ['minDCF log(γ) = -5 no PCA no Znorm','minDCF log(γ) = -5 no PCA Znorm','minDCF log(γ) = -5 PCA=' + str(m) + ' no Znorm','minDCF log(γ) = -5 PCA=' + str(m) + ' Znorm']
        colors = ['b','g','y','c','r']
        #base colors: r, g, b, m, y, c, k, w
        plot.plotDCF([C_values_raw_gamma_1e5,C_values_zNorm_gamma_1e5,C_values_pca_gamma_1e5,C_values_zNormPca_gamma_1e5],[minDcfs_raw_gamma_1e5,minDcfs_zNorm_gamma_1e5,minDcfs_pca_gamma_1e5,minDcfs_zNormPca_gamma_1e5],labels,colors,xlabel='C',title='RBF SVM with $\pi=' + str(prior) + '$')