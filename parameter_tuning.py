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
        minDcfs,_,_ = kfold.K_Fold_LR(DTR,LTR,K=K,classifiers=classifier,lambd=lambd,PCA_Flag=PCA_Flag,M=M,Z_Norm_Flag=Z_Norm_Flag,Dcf_Prior=Dcf_Prior,Calibration_Flag=Calibration_Flag)
        # from classifier parameter from main: first linear, then quadratic
        minDcfs_Linear.append(minDcfs[0])
        minDcfs_Quadratic.append(minDcfs[1])
    
    return minDcfs_Linear,minDcfs_Quadratic
    
def svm_linear_K_C_parameters_testing(DTR,LTR,K=None,k_values=None,C_values=None,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=None,Calibration_Flag=None):  
    #priors = [constants.PRIOR_PROBABILITY]
    plotUtility_list=[]
    #for prior in priors:
    #    print("prior:",prior)        
    for k_value in k_values:
        print("k value : " + str(k_value))
        for C in C_values:
            print("C value : " + str(C))
            minDcf = kfold.K_Fold_SVM_linear(DTR,LTR,K=K,hyperParameter_K=k_value,hyperParameter_C=C,PCA_Flag=PCA_Flag,M=M,Z_Norm_Flag=Z_Norm_Flag,Dcf_Prior=Dcf_Prior,Calibration_Flag=Calibration_Flag)
            plotUtility_list.append(PlotUtility(prior=Dcf_Prior,k=k_value,C=C,minDcf=minDcf))
    
    return plotUtility_list

def svm_kernel_polynomial_K_C_c_d_parameter_testing(DTR,LTR,K=None,k_values=None,C_values=None,c_values=None,d_values=None,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=None,Calibration_Flag=None):
    #priors = [constants.PRIOR_PROBABILITY]
    plotUtility_list=[]
    #for prior in priors:
        #print("prior:",prior)        
    for k_value in k_values:
        print("k value : " + str(k_value))
        for C in C_values:
            print("C value : " + str(C))
            for c in c_values:
                print("c value : " + str(c))
                for d in d_values:
                    print("d value : " + str(d))
                    minDcf = kfold.K_Fold_SVM_kernel_polynomial(DTR,LTR,K=K,hyperParameter_K=k_value,hyperParameter_C=C,hyperParameter_c=c,hyperParameter_d=d,PCA_Flag=PCA_Flag,M=M,Z_Norm_Flag=Z_Norm_Flag,Dcf_Prior=Dcf_Prior,Calibration_Flag=Calibration_Flag)
                    plotUtility_list.append(PlotUtility(prior=Dcf_Prior,k=k_value,C=C,c=c,d=d,minDcf=minDcf))
    
    return plotUtility_list

def svm_kernel_rbf_K_C_gamma_parameter_testing(DTR,LTR,K=None,k_values=None,C_values=None,gamma_values=None,PCA_Flag=None,M=None,Z_Norm_Flag=None,Dcf_Prior=None,Calibration_Flag=None):
    #priors = [constants.PRIOR_PROBABILITY]
    plotUtility_list=[]
    #for prior in priors:
    #    print("prior:",prior)        
    for k_value in k_values:
        print("k value : " + str(k_value))
        for C in C_values:
            print("C value : " + str(C))
            for gamma in gamma_values:
                print("gamma value : " + str(gamma))
                minDcf = kfold.K_Fold_SVM_kernel_rbf(DTR,LTR,K=K,hyperParameter_K=k_value,hyperParameter_C=C,hyperParameter_gamma=gamma,PCA_Flag=PCA_Flag,M=M,Z_Norm_Flag=Z_Norm_Flag,Dcf_Prior=Dcf_Prior,Calibration_Flag=Calibration_Flag)
                plotUtility_list.append(PlotUtility(prior=Dcf_Prior,k=k_value,C=C,gamma=gamma,minDcf=minDcf))
    
    return plotUtility_list