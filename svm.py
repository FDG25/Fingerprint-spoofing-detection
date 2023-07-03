import numpy 
import lr
import sklearn.datasets
import scipy.optimize 
from itertools import repeat

def dualSVM_Objective_gradient(alpha, H):
    # print("Shape of H:", H.shape) #(66,66)
    # print("Shape of alpha:", alpha.shape) (66,)
    #FUNCTION LD_CAPPELLETTO(α) = -JD_CAPPELLETTO(α) THAT WE WANT TO MINIMIZE:
    LD_CAPPELLETTO = (1/2)*numpy.dot(numpy.dot(alpha.T, H), alpha)-numpy.dot(alpha.T, numpy.ones(H.shape[1]))
    #The gradient of -JD_CAPPELLETTO(α) CAN BE COMPUTED AS:
    grad = numpy.dot(H, alpha) - numpy.ones(H.shape[1]) #H.shape[1] coincide con DTR.shape[1]
    return (LD_CAPPELLETTO, grad) #grad HAS SHAPE (n,1) --> in order to be compatible with scipy.optimize.fmin_l_bfgs_b()

def computeDualSVM_solution(DTR, C, H_cappelletto2):
    b = list(repeat((0, C), DTR.shape[1])) #WE have to specify the box constraints 0 ≤ αi ≤ C  #Each element of the list corresponds to a diFFerent optimization variable. In our case, the list should have N elements (DTR.shape[1]), and should be [(0, C), (0, C), ... (0, C)]
    (x, f, d) = scipy.optimize.fmin_l_bfgs_b(dualSVM_Objective_gradient, numpy.zeros(DTR.shape[1]), args=(H_cappelletto2,), bounds=b, factr=1.0) #ESSENDO CHE NON STIAMO SPECIFICANDO approx_grade, DI DEFAULT AVREMO: approx_grad=False --> CI VA BENE VISTO CHE RITORNIAMO GIà IL GRADIENTE CON dualSVM_Objective_gradient! #iprint=1 NON NECESSARIO -->You can control the precision of the L-BFGS solution through the parameter factr . The default value is factr=10000000.0 . Lower values result in more precise solutions (i.e. closer to the optimal solution), but require more iterations. CI VIENE DETTO DI USARE factr=1.0.

    #print("Estimated position of the minimum is: " + str(x))
    #print(x.shape) (66,)
    dual_loss = -f #METTIAMO IL MENO DAVANTI XK NOI ABBIAMO MINIMIZZATO -JD_CAPPELLETTO(α) X POTER UTILIZZARE scipy.optimize.fmin_l_bfgs_b, MA LA NOSTRA FUNZIONE DI PARTENZA è JD_CAPPELLETTO(α)
    #print(f"DUAL LOSS --> The objective value at the minimum is: {round(dual_loss, 6)}") 
    #print(d)
    return dual_loss, x

def computeScores(w_CAPPELLETTO_asterisco, DTE_cappelletto, LTE, C):
    S = numpy.dot(w_CAPPELLETTO_asterisco, DTE_cappelletto)
    #print(S.shape) #(34,) --> 1 score per ogni test sample!
    
    LP = [] #LP is the array of predicted labels for the test sample
    for i in range(0,S.size): #per ogni elemento nel test set
        if S[i] > 0:  
            LP.append(1) #IF S[i] > 0 --> LP[i] = 1 (PREDICIAMO LA CLASSE 1 per il campione i-esimo)
        else:
            LP.append(0) #IF S[i] <= 0, PREDICIAMO LA CLASSE 0
    #SICCOME LTE NON è STATO MODIFICATO ASSEGNANDO -1 AL POSTO DEGLI 0, CI BASTA FARE:
    nCorrectPredictions = lr.computeNumCorrectPredictionsDiscriminative(numpy.array(LP), LTE)
    acc = nCorrectPredictions/LTE.size
    err = 1 - acc
    #print("Accuracy: " + str(round(acc*100, 1)) + "%")
    #print("Error rate: " + str(round(err*100, 1)) + "%")
    return S,nCorrectPredictions


def computePrimalSVM_solution_through_dual_SVM_formulation(DTR_cappelletto, LTR, w_CAPPELLETTO_asterisco, C):
    first_term = 1/2*(numpy.linalg.norm(w_CAPPELLETTO_asterisco)**2)
    max = numpy.zeros(LTR.size)
    for i in range(LTR.size):
        vett = [0, 1-LTR[i]*(numpy.dot(w_CAPPELLETTO_asterisco.T, DTR_cappelletto[:, i]))]
        max[i] = vett[numpy.argmax(vett)]
    second_term = C*numpy.sum(max)
    loss = first_term + second_term
    return loss


def computeDualityGap(primary_loss, dual_loss):
    return (primary_loss - dual_loss)


######################################
#KERNEL SVM
def computeScoresNonLinear(x, DTR, LTR, LTE, kernelFunctionTest):
    S = numpy.sum(numpy.dot((x*LTR).reshape(1, DTR.shape[1]), kernelFunctionTest), axis=0)
    #print(S.shape) #(34,) --> 1 score per ogni test sample!
    
    LP = [] #LP is the array of predicted labels for the test sample
    for i in range(0,S.size): #per ogni elemento nel test set
        if S[i] > 0:  
            LP.append(1) #IF S[i] > 0 --> LP[i] = 1 (PREDICIAMO LA CLASSE 1 per il campione i-esimo)
        else:
            LP.append(0) #IF S[i] <= 0, PREDICIAMO LA CLASSE 0
    #SICCOME LTE NON è STATO MODIFICATO ASSEGNANDO -1 AL POSTO DEGLI 0, CI BASTA FARE:
    nCorrectPredictions = lr.computeNumCorrectPredictionsDiscriminative(numpy.array(LP), LTE)
    acc = nCorrectPredictions/LTE.size
    err = 1 - acc
    #print("Accuracy: " + str(round(acc*100, 1)) + "%")
    #print("Error rate: " + str(round(err*100, 1)) + "%")
    #print()
    return S,nCorrectPredictions



def linear_svm(DTR,LTR,DTE,LTE,k_value,C):
    #################################################
    
    #INNANZITUTTO CALCOLIAMO tutti gli zi
    for i in range(0,LTR.size):
        if LTR[i] == 0:
            # class 0, assign z_i = -1
            LTR[i] = -1
        else:
            # class 1, assign z_i = 1
            LTR[i] = 1

    # compute DTR_cappelletto
    DTR_cappelletto = numpy.vstack((DTR, numpy.zeros(DTR.shape[1])+k_value))
    #print(DTR_cappelletto.shape) #(5, 66) #COME D MA OGNI CAMPIONE OLTRE AI VALORI DELLE 4 FEATURES HA UN 1 ALLA FINE
    #for i in range(0,DTR.shape[1]):
    #    print(DTR_cappelletto[:,i])

    # compute H_cappelletto --> 2 MODI PER FARLO
    #1)PER TROVARE H_cappelletto POSSIAMO USARE DEI for loops:
    # H_cappelletto1 = numpy.zeros((DTR.shape[1], DTR.shape[1]))
    # for i in range(0,DTR.shape[1]):
    #     for j in range(0,DTR.shape[1]):
    #         H_cappelletto1[i, j] = LTR[i] * LTR[j] * numpy.dot((DTR_cappelletto[:, i]).transpose(), DTR_cappelletto[:, j])
    # print(H_cappelletto1)
        
    #2)WE can speed up computations exploiting numpy.dot to compute the matrix G_cappelletto
    #from DTR_cappelletto in a single call, and broadcasting to compute H_cappelletto from G_cappelletto:
    Gij = numpy.dot(DTR_cappelletto.T, DTR_cappelletto)
    # To compute zi*zj WE need to reshape LTR as a matrix with one column/row and then do the dot product
    zizj = numpy.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    H_cappelletto2 = zizj*Gij
    #print(H_cappelletto2)
            

    dual_loss, x = computeDualSVM_solution(DTR, C, H_cappelletto2)

    #Once WE have computed the dual solution, WE can recover the primal solution through 
    #w_CAPPELLETTO* = SOMMATORIA DA i=1 A N di (αizixi_cappelletto) --> GLI αi SONO CONTENUTI IN x
    w_CAPPELLETTO_asterisco = numpy.sum((x*LTR).reshape(1, DTR.shape[1])*DTR_cappelletto, axis=1) #.reshape(1, DTR.shape[1]) SI PUò OMETTERE!
    primal_loss = computePrimalSVM_solution_through_dual_SVM_formulation(DTR_cappelletto, LTR, w_CAPPELLETTO_asterisco, C)
    #print(f"PRIMAL LOSS --> {round(primal_loss, 6)}") 
    duality_gap = computeDualityGap(primal_loss, dual_loss)
    #print("DUALITY GAP --> " + str(duality_gap)) 

    DTE_cappelletto = numpy.vstack((DTE, numpy.zeros(DTE.shape[1])+k_value))
    #TO CLASSIFY A PATTERN 1)WE CAN extract the terms w∗, b∗ from w_CAPPELLETTO*
    #and then compute the solution as w∗T xt + b∗ OR 2)WE CAN DIRECTLY compute the score DOING w_CAPPELLETTO*Txt_CAPPELLETTO
    scores,correct_predictions = computeScores(w_CAPPELLETTO_asterisco, DTE_cappelletto, LTE, C) #UTILIZZIAMO LA DUAL SVM SOLUTION PER CALCOLARE GLI SCORE
    return (scores,correct_predictions)



def kernel_svm_polynomial(DTR,LTR,DTE,LTE,k_value,C,c,d):
    #################################################
    
    #INNANZITUTTO CALCOLIAMO tutti gli zi
    for i in range(0,LTR.size):
        if LTR[i] == 0:
            # class 0, assign z_i = -1
            LTR[i] = -1
        else:
            # class 1, assign z_i = 1
            LTR[i] = 1
    #################################################
    #2)KERNEL SVM (non-linear SVM) --> SVMs allow for non-linear classification through an implicit expansion of the features in a higher dimensional space.
    #a)Polynomial kernel of degree d
    #The choice of the kernel and of its hyper-parameters (e.g. c and γ) can also be made through cross-validation.
    # NON USIAMO DTR_cappelletto, MA DTR NORMALE!
    # compute H_cappelletto --> 2 MODI PER FARLO
    #NON USIAMO Gij --> we want to compute the scalar product between the expanded features k(x1, x2) = φ(x1)T φ(x2). Function k is called kernel function.
    # To compute zi*zj WE need to reshape LTR as a matrix with one column/row and then do the dot product
    zizj = numpy.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    polynomial_kernel = (numpy.dot(DTR.T, DTR)+c)**d + k_value**2
    H_cappelletto2 = zizj*polynomial_kernel
    #print(H_cappelletto2)
    #print("Polynomial kernel of degree d")  
    #print("K = " + str(k_value) + "; " + "C = " + str(C) + "; " + "d = " + str(d) + "; " + "c = " + str(c))
    dual_loss, x = computeDualSVM_solution(DTR, C, H_cappelletto2)
    #In contrast with linear SVM, IN THIS CASE we are not able to compute the primal solution and its cost
    #NON USIAMO w_CAPPELLETTO_asterisco
    #NON USIAMO NEANCHE DTE_cappelletto
    polynomial_kernel_TEST = ((numpy.dot(DTR.T, DTE)+c)**d + k_value) #qui K non viene elevato al quadrato
    scores,correct_predictions = computeScoresNonLinear(x, DTR, LTR, LTE, polynomial_kernel_TEST) #UTILIZZIAMO LA DUAL SVM SOLUTION PER CALCOLARE GLI SCORE
    return (scores,correct_predictions)

def kernel_svm_radial(DTR,LTR,DTE,LTE,k_value,C,gamma):
    #################################################
    
    #INNANZITUTTO CALCOLIAMO tutti gli zi
    for i in range(0,LTR.size):
        if LTR[i] == 0:
            # class 0, assign z_i = -1
            LTR[i] = -1
        else:
            # class 1, assign z_i = 1
            LTR[i] = 1
    #b)Radial Basis Function kernel
    # NON USIAMO DTR_cappelletto, MA DTR NORMALE!
    # compute H_cappelletto --> 2 MODI PER FARLO
    #NON USIAMO Gij --> we want to compute the scalar product between the expanded features k(x1, x2) = φ(x1)T φ(x2). Function k is called kernel function.
    # To compute zi*zj WE need to reshape LTR as a matrix with one column/row and then do the dot product
    zizj = numpy.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    rbf_kernel = numpy.zeros((DTR.shape[1], DTR.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTR.shape[1]):
            rbf_kernel[i,j] = numpy.exp(-gamma*(numpy.linalg.norm(DTR[:, i]-DTR[:, j])**2))+k_value**2
    # To compute zi*zj I need to reshape LTR as a matrix with one column/row
    # and then do the dot product
    zizj = numpy.dot(LTR.reshape(LTR.size, 1), LTR.reshape(1, LTR.size))
    H_cappelletto2 = zizj*rbf_kernel
    #print(H_cappelletto2)
    #print("Radial Basis Function kernel")
    #print("K = " + str(k_value) + "; " + "C = " + str(C) + "; " + "gamma = " + str(gamma))
    dual_loss, x = computeDualSVM_solution(DTR, C, H_cappelletto2)
    #In contrast with linear SVM, IN THIS CASE we are not able to compute the primal solution and its cost
    #NON USIAMO w_CAPPELLETTO_asterisco
    #NON USIAMO NEANCHE DTE_cappelletto
    rbf_kernel = numpy.zeros((DTR.shape[1], DTE.shape[1]))
    for i in range(DTR.shape[1]):
        for j in range(DTE.shape[1]):
            rbf_kernel[i,j] = numpy.exp(-gamma*(numpy.linalg.norm(DTR[:, i]-DTE[:, j])**2))+k_value**2
    scores,correct_predictions = computeScoresNonLinear(x, DTR, LTR, LTE, rbf_kernel) #UTILIZZIAMO LA DUAL SVM SOLUTION PER CALCOLARE GLI SCORE
    return (scores,correct_predictions)