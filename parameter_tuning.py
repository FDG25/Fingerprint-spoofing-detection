import constants
import kfold
import plot
from plot_utility import PlotUtility
import numpy

def lr_lambda_parameter_testing(DTR,LTR,K=None,lambda_values=None,classifier=None,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=None,Calibration_Flag=None):    
    #priors = [0.5, 0.9, 0.1]
    #priors = [constants.PRIOR_PROBABILITY]
    minDcfs_Linear = []
    minDcfs_Quadratic = []
    #for i in range(len(priors)):
    #    print("prior:",priors[i])        
    for lambd in lambda_values:
        print("lambda value : " + str(lambd))
        minDcfs = kfold.K_Fold_LR(DTR,LTR,K=5,classifiers=classifier,lambd=lambd,PCA_Flag=PCA_Flag,M=M,Z_Norm_Flag=Z_Norm_Flag,Dcf_Prior=Dcf_Prior,Calibration_Flag=Calibration_Flag)
        # from classifier parameter from main: first linear, then quadratic
        minDcfs_Linear.append(minDcfs[0])
        #minDcfs_Quadratic.append(minDcfs[1])
    
    return minDcfs_Linear,minDcfs_Quadratic
    
def svm_linear_K_C_parameters_testing(DTR,LTR,K=None,k_values=None,C_values=None,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=None,Calibration_Flag=None):  
    priors = [constants.PRIOR_PROBABILITY]
    plotUtility_list=[]
    for prior in priors:
        print("prior:",prior)        
        for k_value in k_values:
            print("k value : " + str(k_value))
            for C in C_values:
                print("C value : " + str(C))
                minDcf = kfold.K_Fold_SVM_linear(DTR,LTR,K=K,hyperParameter_K=k_value,hyperParameter_C=C,PCA_Flag=PCA_Flag,M=M,Z_Norm_Flag=Z_Norm_Flag,Dcf_Prior=Dcf_Prior,Calibration_Flag=Calibration_Flag)
                plotUtility_list.append(PlotUtility(prior=prior,k=k_value,C=C,minDcf=minDcf))
    
    return plotUtility_list

def svm_kernel_polynomial_K_C_c_d_parameter_testing(DTR,LTR,k_values,C_values,c_values,d_values):
    priors = [constants.PRIOR_PROBABILITY]
    plotUtility_list=[]
    for prior in priors:
        print("prior:",prior)        
        for k_value in k_values:
            print("k value : " + str(k_value))
            for C in C_values:
                print("C value : " + str(C))
                for c in c_values:
                    print("c value : " + str(c))
                    for d in d_values:
                        print("d value : " + str(d))
                        minDcf = kfold.K_Fold_SVM_kernel_polynomial(DTR,LTR,K=5,hyperParameter_K=k_value,hyperParameter_C=C,hyperParameter_c=c,hyperParameter_d=d)
                        plotUtility_list.append(PlotUtility(prior=prior,k=k_value,C=C,c=c,d=d,minDcf=minDcf))
    
    # ----- PLOT FOR K = 1 ------
    k_1_c_0_d_2 = list(filter(lambda PlotElement: PlotElement.is_k(1) and PlotElement.is_c(0) and PlotElement.is_d(2), plotUtility_list))
    minDcfs_k_1_c_0_d_2 = [PlotElement.getminDcf() for PlotElement in k_1_c_0_d_2]
    C_values_k_1_c_0_d_2 = [PlotElement.getC() for PlotElement in k_1_c_0_d_2]

    k_1_c_0_d_4 = list(filter(lambda PlotElement: PlotElement.is_k(1) and PlotElement.is_c(0) and PlotElement.is_d(4), plotUtility_list))
    minDcfs_k_1_c_0_d_4 = [PlotElement.getminDcf() for PlotElement in k_1_c_0_d_4]
    C_values_k_1_c_0_d_4 = [PlotElement.getC() for PlotElement in k_1_c_0_d_4]

    k_1_c_1_d_2 = list(filter(lambda PlotElement: PlotElement.is_k(1) and PlotElement.is_c(1) and PlotElement.is_d(2), plotUtility_list))
    minDcfs_k_1_c_1_d_2 = [PlotElement.getminDcf() for PlotElement in k_1_c_1_d_2]
    C_values_k_1_c_1_d_2 = [PlotElement.getC() for PlotElement in k_1_c_1_d_2]

    k_1_c_1_d_4 = list(filter(lambda PlotElement: PlotElement.is_k(1) and PlotElement.is_c(1) and PlotElement.is_d(4), plotUtility_list))
    minDcfs_k_1_c_1_d_4 = [PlotElement.getminDcf() for PlotElement in k_1_c_1_d_4]
    C_values_k_1_c_1_d_4 = [PlotElement.getC() for PlotElement in k_1_c_1_d_4]

    labels = ['K = 1,c = 0,D = 2','K = 1,c = 0,D = 4','K = 1,c = 1,D = 2','K = 1,c = 1,D = 4']
    colors = ['b','g','y','c']
    #base colors: r, g, b, m, y, c, k, w
    plot.plotDCF([C_values_k_1_c_0_d_2,C_values_k_1_c_0_d_4,C_values_k_1_c_1_d_2,C_values_k_1_c_1_d_4],[minDcfs_k_1_c_0_d_2,minDcfs_k_1_c_0_d_4,minDcfs_k_1_c_1_d_2,minDcfs_k_1_c_1_d_4],labels,colors,'C')


    # ----- PLOT FOR K = 10 ------
    k_10_c_0_d_2 = list(filter(lambda PlotElement: PlotElement.is_k(10) and PlotElement.is_c(0) and PlotElement.is_d(2), plotUtility_list))
    minDcfs_k_10_c_0_d_2 = [PlotElement.getminDcf() for PlotElement in k_10_c_0_d_2]
    C_values_k_10_c_0_d_2 = [PlotElement.getC() for PlotElement in k_10_c_0_d_2]

    k_10_c_0_d_4 = list(filter(lambda PlotElement: PlotElement.is_k(10) and PlotElement.is_c(0) and PlotElement.is_d(4), plotUtility_list))
    minDcfs_k_10_c_0_d_4 = [PlotElement.getminDcf() for PlotElement in k_10_c_0_d_4]
    C_values_k_10_c_0_d_4 = [PlotElement.getC() for PlotElement in k_10_c_0_d_4]

    k_10_c_1_d_2 = list(filter(lambda PlotElement: PlotElement.is_k(10) and PlotElement.is_c(1) and PlotElement.is_d(2), plotUtility_list))
    minDcfs_k_10_c_1_d_2 = [PlotElement.getminDcf() for PlotElement in k_10_c_1_d_2]
    C_values_k_10_c_1_d_2 = [PlotElement.getC() for PlotElement in k_10_c_1_d_2]

    k_10_c_1_d_4 = list(filter(lambda PlotElement: PlotElement.is_k(10) and PlotElement.is_c(1) and PlotElement.is_d(4), plotUtility_list))
    minDcfs_k_10_c_1_d_4 = [PlotElement.getminDcf() for PlotElement in k_10_c_1_d_4]
    C_values_k_10_c_1_d_4 = [PlotElement.getC() for PlotElement in k_10_c_1_d_4]

    labels = ['K = 10,c = 0,D = 2','K = 10,c = 0,D = 4','K = 10,c = 1,D = 2','K = 10,c = 1,D = 4']
    colors = ['b','g','y','c']
    #base colors: r, g, b, m, y, c, k, w
    plot.plotDCF([C_values_k_10_c_0_d_2,C_values_k_10_c_0_d_4,C_values_k_10_c_1_d_2,C_values_k_10_c_1_d_4],[minDcfs_k_10_c_0_d_2,minDcfs_k_10_c_0_d_4,minDcfs_k_10_c_1_d_2,minDcfs_k_10_c_1_d_4],labels,colors,'C')

    
    # ----- PLOT FOR K = 100 ------
    k_100_c_0_d_2 = list(filter(lambda PlotElement: PlotElement.is_k(100) and PlotElement.is_c(0) and PlotElement.is_d(2), plotUtility_list))
    minDcfs_k_100_c_0_d_2 = [PlotElement.getminDcf() for PlotElement in k_100_c_0_d_2]
    C_values_k_100_c_0_d_2 = [PlotElement.getC() for PlotElement in k_100_c_0_d_2]

    k_100_c_0_d_4 = list(filter(lambda PlotElement: PlotElement.is_k(100) and PlotElement.is_c(0) and PlotElement.is_d(4), plotUtility_list))
    minDcfs_k_100_c_0_d_4 = [PlotElement.getminDcf() for PlotElement in k_100_c_0_d_4]
    C_values_k_100_c_0_d_4 = [PlotElement.getC() for PlotElement in k_100_c_0_d_4]

    k_100_c_1_d_2 = list(filter(lambda PlotElement: PlotElement.is_k(100) and PlotElement.is_c(1) and PlotElement.is_d(2), plotUtility_list))
    minDcfs_k_100_c_1_d_2 = [PlotElement.getminDcf() for PlotElement in k_100_c_1_d_2]
    C_values_k_100_c_1_d_2 = [PlotElement.getC() for PlotElement in k_100_c_1_d_2]

    k_100_c_1_d_4 = list(filter(lambda PlotElement: PlotElement.is_k(100) and PlotElement.is_c(1) and PlotElement.is_d(4), plotUtility_list))
    minDcfs_k_100_c_1_d_4 = [PlotElement.getminDcf() for PlotElement in k_100_c_1_d_4]
    C_values_k_100_c_1_d_4 = [PlotElement.getC() for PlotElement in k_100_c_1_d_4]

    labels = ['K = 100,c = 0,D = 2','K = 100,c = 0,D = 4','K = 100,c = 1,D = 2','K = 100,c = 1,D = 4']
    colors = ['b','g','y','c']
    #base colors: r, g, b, m, y, c, k, w
    plot.plotDCF([C_values_k_100_c_0_d_2,C_values_k_100_c_0_d_4,C_values_k_100_c_1_d_2,C_values_k_100_c_1_d_4],[minDcfs_k_100_c_0_d_2,minDcfs_k_100_c_0_d_4,minDcfs_k_100_c_1_d_2,minDcfs_k_100_c_1_d_4],labels,colors,'C')

def svm_kernel_rbf_K_C_gamma_parameter_testing(DTR,LTR,k_values,C_values,gamma_values):
    priors = [constants.PRIOR_PROBABILITY]
    plotUtility_list=[]
    for prior in priors:
        print("prior:",prior)        
        for k_value in k_values:
            print("k value : " + str(k_value))
            for C in C_values:
                print("C value : " + str(C))
                for gamma in gamma_values:
                    print("gamma value : " + str(gamma))
                    minDcf = kfold.K_Fold_SVM_kernel_rbf(DTR,LTR,K=5,hyperParameter_K=k_value,hyperParameter_C=C,hyperParameter_gamma=gamma)
                    plotUtility_list.append(PlotUtility(prior=prior,k=k_value,C=C,gamma=gamma,minDcf=minDcf))
    
    # ---- PLOT FOR K = 0 -----
    k_0_gamma_1e1 = list(filter(lambda PlotElement: PlotElement.is_k(0) and PlotElement.is_gamma(1/numpy.exp(1)), plotUtility_list))
    minDcfs_k_0_gamma_1e1 = [PlotElement.getminDcf() for PlotElement in k_0_gamma_1e1]
    C_values_k_0_gamma_1e1 = [PlotElement.getC() for PlotElement in k_0_gamma_1e1]

    k_0_gamma_1e2 = list(filter(lambda PlotElement: PlotElement.is_k(0) and PlotElement.is_gamma(1/numpy.exp(2)), plotUtility_list))
    minDcfs_k_0_gamma_1e2 = [PlotElement.getminDcf() for PlotElement in k_0_gamma_1e2]
    C_values_k_0_gamma_1e2 = [PlotElement.getC() for PlotElement in k_0_gamma_1e2]

    k_0_gamma_1e3 = list(filter(lambda PlotElement: PlotElement.is_k(0) and PlotElement.is_gamma(1/numpy.exp(3)), plotUtility_list))
    minDcfs_k_0_gamma_1e3 = [PlotElement.getminDcf() for PlotElement in k_0_gamma_1e3]
    C_values_k_0_gamma_1e3 = [PlotElement.getC() for PlotElement in k_0_gamma_1e3]

    k_0_gamma_1e4 = list(filter(lambda PlotElement: PlotElement.is_k(0) and PlotElement.is_gamma(1/numpy.exp(4)), plotUtility_list))
    minDcfs_k_0_gamma_1e4 = [PlotElement.getminDcf() for PlotElement in k_0_gamma_1e4]
    C_values_k_0_gamma_1e4 = [PlotElement.getC() for PlotElement in k_0_gamma_1e4]

    k_0_gamma_1e5 = list(filter(lambda PlotElement: PlotElement.is_k(0) and PlotElement.is_gamma(1/numpy.exp(5)), plotUtility_list))
    minDcfs_k_0_gamma_1e5 = [PlotElement.getminDcf() for PlotElement in k_0_gamma_1e5]
    C_values_k_0_gamma_1e5 = [PlotElement.getC() for PlotElement in k_0_gamma_1e5]

    labels = ['K = 0,log(γ) = -1','K = 0,log(γ) = -2','K = 0,log(γ) = -3','K = 0,log(γ) = -4','K = 0,log(γ) = -5']
    colors = ['b','g','y','c','r']
    #base colors: r, g, b, m, y, c, k, w
    plot.plotDCF([C_values_k_0_gamma_1e1,C_values_k_0_gamma_1e2,C_values_k_0_gamma_1e3,C_values_k_0_gamma_1e4,C_values_k_0_gamma_1e5],[minDcfs_k_0_gamma_1e1,minDcfs_k_0_gamma_1e2,minDcfs_k_0_gamma_1e3,minDcfs_k_0_gamma_1e4,minDcfs_k_0_gamma_1e5],labels,colors,'C')


    # ---- PLOT FOR K = 1 -----
    k_1_gamma_1e1 = list(filter(lambda PlotElement: PlotElement.is_k(1) and PlotElement.is_gamma(1/numpy.exp(1)), plotUtility_list))
    minDcfs_k_1_gamma_1e1 = [PlotElement.getminDcf() for PlotElement in k_1_gamma_1e1]
    C_values_k_1_gamma_1e1 = [PlotElement.getC() for PlotElement in k_1_gamma_1e1]

    k_1_gamma_1e2 = list(filter(lambda PlotElement: PlotElement.is_k(1) and PlotElement.is_gamma(1/numpy.exp(2)), plotUtility_list))
    minDcfs_k_1_gamma_1e2 = [PlotElement.getminDcf() for PlotElement in k_1_gamma_1e2]
    C_values_k_1_gamma_1e2 = [PlotElement.getC() for PlotElement in k_1_gamma_1e2]

    k_1_gamma_1e3 = list(filter(lambda PlotElement: PlotElement.is_k(1) and PlotElement.is_gamma(1/numpy.exp(3)), plotUtility_list))
    minDcfs_k_1_gamma_1e3 = [PlotElement.getminDcf() for PlotElement in k_1_gamma_1e3]
    C_values_k_1_gamma_1e3 = [PlotElement.getC() for PlotElement in k_1_gamma_1e3]

    k_1_gamma_1e4 = list(filter(lambda PlotElement: PlotElement.is_k(1) and PlotElement.is_gamma(1/numpy.exp(4)), plotUtility_list))
    minDcfs_k_1_gamma_1e4 = [PlotElement.getminDcf() for PlotElement in k_1_gamma_1e4]
    C_values_k_1_gamma_1e4 = [PlotElement.getC() for PlotElement in k_1_gamma_1e4]

    k_1_gamma_1e5 = list(filter(lambda PlotElement: PlotElement.is_k(1) and PlotElement.is_gamma(1/numpy.exp(5)), plotUtility_list))
    minDcfs_k_1_gamma_1e5 = [PlotElement.getminDcf() for PlotElement in k_1_gamma_1e5]
    C_values_k_1_gamma_1e5 = [PlotElement.getC() for PlotElement in k_1_gamma_1e5]

    labels = ['K = 1,log(γ) = -1','K = 1,log(γ) = -2','K = 1,log(γ) = -3','K = 1,log(γ) = -4','K = 1,log(γ) = -5']
    colors = ['b','g','y','c','r']
    #base colors: r, g, b, m, y, c, k, w
    plot.plotDCF([C_values_k_1_gamma_1e1,C_values_k_1_gamma_1e2,C_values_k_1_gamma_1e3,C_values_k_1_gamma_1e4,C_values_k_1_gamma_1e5],[minDcfs_k_1_gamma_1e1,minDcfs_k_1_gamma_1e2,minDcfs_k_1_gamma_1e3,minDcfs_k_1_gamma_1e4,minDcfs_k_1_gamma_1e5],labels,colors,'C')