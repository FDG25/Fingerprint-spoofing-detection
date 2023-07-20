import main
import numpy
import matplotlib.pyplot as plt

#Confusion matrices are a tool to visualize the number of samples predicted as class i and belonging to
#class j. A confusion matrix is a K × K matrix whose elements Mi,j represent the number of samples
#belonging to class j that are predicted as class i.
def confusion_matrix(y_true, y_pred, num_classes=2): #NUOVA FUNZIONE
    cm = numpy.zeros((num_classes, num_classes), dtype=int)
    for i in range(num_classes):
        for j in range(num_classes):
            cm[i][j] = numpy.sum((y_pred == i) & (y_true == j))
    return cm


#FNR and FPR are the false positive and false negative rates
def computeFnr_Fpr(confusion_matrix):
    FNR = confusion_matrix[0][1]/(confusion_matrix[0][1] + confusion_matrix[1][1]) #FNR = FN/(FN+TP) 
    FPR = confusion_matrix[1][0]/(confusion_matrix[1][0] + confusion_matrix[0][0]) #FPR = FP/(FP+TN)
    return FNR, FPR

def computeDCF(confusion_matrix, pi, Cfn, Cfp): 
    FNR,FPR = computeFnr_Fpr(confusion_matrix)
    # the returned value is the detection cost function
    return pi * Cfn * FNR + (1-pi) * Cfp * FPR

def computeDCFNormalized(confusion_matrix, pi, Cfn, Cfp): 
    FNR = confusion_matrix[0][1]/(confusion_matrix[0][1] + confusion_matrix[1][1])
    FPR = confusion_matrix[1][0]/(confusion_matrix[0][0] + confusion_matrix[1][0])
    factor = 0
    if pi * Cfn > (1-pi) * Cfp:
        factor = (1-pi) * Cfp
    else:
        factor = pi * Cfn
    return computeDCF(confusion_matrix, pi, Cfn, Cfp)/factor

def computeMinDCF(pi,Cfn,Cfp,llrs,labels):
    res = numpy.zeros(llrs.shape)
    llrs_S=numpy.sort(llrs)
    for i in range(0,llrs.size):
        res[i],_,_ = computeOptimalDecisionBinaryBayesPlot(pi,Cfn,Cfp,llrs,labels,llrs_S[i])
    return numpy.min(res)

def computePredictedLabels(x,threshold):
    if x > threshold:
        return 1
    else:
        return 0

# computes optimal Bayes decisions for different priors and costs starting from binary log-likelihood ratios
def computeOptimalDecisionBinary(pi,Cfn,Cfp,llrs,labels,threshold=None):
    EFFECTIVE_PRIOR = (pi * Cfn)/((1-pi) * Cfp)
    isOptimal = 0
    if threshold is None:
        threshold = -numpy.log(EFFECTIVE_PRIOR)
        isOptimal = 1
    
    LP3 = numpy.vectorize(computePredictedLabels)(llrs, threshold)
    cm = confusion_matrix(labels, LP3, num_classes=2)
    if isOptimal == 1:
        print("pi: " + str(pi) + "; Cfn: " + str(Cfn) + "; Cfp: " + str(Cfp))
        print(cm)
    DCF = computeDCF(cm,pi,Cfn,Cfp)
    if isOptimal == 1:
        print("DCF: " + str(round(DCF,3)))
    N_DCF = computeDCFNormalized(cm,pi,Cfn,Cfp)
    if isOptimal == 1:
        print("Normalized DCF: " + str(round(N_DCF,3)))
        print()
    FNR,FPR = computeFnr_Fpr(cm)
    TPR = 1 - FNR
    return N_DCF,FPR,TPR

def computeOptimalDecisionBinaryBayesPlot(pi,Cfn,Cfp,llrs,labels,threshold=None):
    EFFECTIVE_PRIOR = (pi * Cfn)/((1-pi) * Cfp)
    if threshold is None:
        threshold = -numpy.log(EFFECTIVE_PRIOR)
    
    LP3 = numpy.vectorize(computePredictedLabels)(llrs, threshold)
    cm = confusion_matrix(labels, LP3, num_classes=2)
    N_DCF = computeDCFNormalized(cm,pi,Cfn,Cfp)
    FNR,FPR = computeFnr_Fpr(cm)
    TPR = 1 - FNR
    return N_DCF,FPR,TPR

def plotROCcurve(FPR, TPR, title):
    # Function used to plot TPR(FPR)
    plt.figure()
    plt.grid(linestyle='--')
    plt.plot(FPR, TPR, linewidth=2)
    plt.xlabel("FPR") 
    plt.ylabel("TPR")
    plt.title(title)
    plt.show()

def bayesErrorPlotMerged(dcf1, mindcf1, dcf2, mindcf2, effPriorLogOdds): #dcf is the array containing the DCF values, and mindcf is the array containing the minimum DCF values
    plt.figure()
    plt.plot(effPriorLogOdds, dcf1, label='DCF (ε=0.001)', color='r')
    plt.plot(effPriorLogOdds, mindcf1, label='min DCF (ε=0.001)', color='b')
    plt.plot(effPriorLogOdds, dcf2, label='DCF (ε=1)', color='y')
    plt.plot(effPriorLogOdds, mindcf2, label='min DCF (ε=1)', color='g')
    plt.xlabel("prior log-odds")
    plt.ylabel("DCF value")
    plt.legend(loc='lower left')
    plt.title("Bayes Error Plot")
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.show()
