import numpy
import scipy 
import main
import constants
import normalization

#   --------------      LOGISTIC REGRESSION MODEL   --------------




'''
# non weighted without gradient version of the model not used
def logreg_obj_wrap(DTR, LTR, lambd):
    def logreg_obj(v):
        w, b = v[0:-1], v[-1]
        n = DTR.shape[1]
        first_term = (lambd/2) * (numpy.linalg.norm(w)**2)
        second_term = 1/n
        third_term = 0
        for i in range(0,n):
            c_i = LTR[i]
            z_i = 2*c_i-1
            x_i = DTR[:,i]
            third_term += numpy.logaddexp(0,-z_i*((numpy.dot(w.T,x_i))+b))
        return first_term + second_term * third_term
    return logreg_obj
'''

'''
# to use with approx_grad = true, but way too slower with quadratic model so the gradient version is used
def logreg_obj_wrap_weighted(DTR, LTR, lambd):
    def logreg_obj_weighted(v):
        w, b = v[0:-1], v[-1]
        n = DTR.shape[1]
        first_term = (lambd/2) * (numpy.linalg.norm(w)**2)
        DP0,DP1 = main.getClassMatrix(DTR,LTR)
        #app_prior = constants.EFFECTIVE_PRIOR
        app_prior = constants.PRIOR_PROBABILITY
        weight_0 = (1-app_prior)/DP0.shape[1]
        weight_1 = app_prior/DP1.shape[1]
        loss_0 = 0
        loss_1 = 0
        for i in range(0,n):
            c_i = LTR[i]
            z_i = 2*c_i-1
            x_i = DTR[:,i]
            if z_i == -1:
                # class 0
                loss_0 += numpy.logaddexp(0,-z_i*((numpy.dot(w.T,x_i))+b))
            else:
                # class 1
                loss_1 += numpy.logaddexp(0,-z_i*((numpy.dot(w.T,x_i))+b))
        return first_term + weight_1 * loss_1 + weight_0 * loss_0
    return logreg_obj_weighted
'''

def computeNumCorrectPredictionsDiscriminative(LP,LTE):
    bool_val = numpy.array(LP==LTE)
    #n_samples = LTE.size
    #acc = numpy.sum(bool_val)/n_samples
    #err = 1 - acc
    return numpy.sum(bool_val)

def computeScores(DTE,LTE,v):
    w, b = v[0:-1], v[-1]
    n = DTE.shape[1]
    LP = []
    llrs = []
    for i in range(0,n):
        x_t = DTE[:,i]
        s_i = numpy.dot(w.T,x_t)
        s_i+=b
        llrs.append(s_i)
        if s_i > 0:
            LP.append(1)
        else:
            LP.append(0)
    nCorrectPredictions = computeNumCorrectPredictionsDiscriminative(numpy.array(LP),LTE)
    #acc = nCorrectPredictions/LTE.size
    #err = 1 - acc
    #print("Accuracy: " + str(round(acc*100, 1)) + "%")
    #print("Error rate: " + str(round(err*100, 1)) + "%")
    return numpy.array(llrs),nCorrectPredictions
'''
def LogisticRegression(DTR,LTR,DTE,LTE):
    lambd = constants.LAMDBA
    #lambda_values = [0.000001, 0.001, 0.1, 1.0] 
    #for lambd in lambda_values: 
    logreg_obj = logreg_obj_wrap(DTR, LTR, lambd) 
    (x, f, d) = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTR.shape[0] + 1), approx_grad=True)
    #print("Lambda=" + str(lambd)) 
    #print(f"The objective value at the minimum (J(w*,b*)) is: {round(f, 7)}") 
    return computeScores(DTE, LTE, x) 
'''
def LogisticRegressionWeighted(DTR,LTR,DTE,LTE,lambd):
    
    # ---------------- NORMALIZING APPROACHES ---------------------
    #T_DTR = normalization.zNormalizingData(DTR)
    #T_DTE = normalization.zNormalizingData(DTE)
    #C_DTR = normalization.centeringData(DTR)
    #C_DTE = normalization.centeringData(DTE)
    #W_DTR = normalization.whiteningData(C_DTR)
    #W_DTE = normalization.whiteningData(C_DTE)
    #L2_DTR = normalization.l2NormalizingData(W_DTR)
    #L2_DTE = normalization.l2NormalizingData(W_DTE)

    # ----- WITHOUT GRADIENT CALL   -------
    #logreg_obj_weighted = logreg_obj_wrap_weighted(DTR, LTR, lambd) 
    #(x, f, d) = scipy.optimize.fmin_l_bfgs_b(logreg_obj_weighted, numpy.zeros(DTR.shape[0] + 1), approx_grad=True)
    # ----- WITH GRADIENT, SO WITHOUT APPROX_GRAD=TRUE  ------
    logreg_obj_quadratic_weighted_gradient = logreg_obj_wrap_weighted_gradient(DTR, LTR, lambd) 
    (x, f, d) = scipy.optimize.fmin_l_bfgs_b(logreg_obj_quadratic_weighted_gradient, numpy.zeros(DTR.shape[0] + 1))
    #print("Lambda=" + str(lambd)) 
    #print(f"The objective value at the minimum (J(w*,b*)) is: {round(f, 7)}") 
    return computeScores(DTE, LTE, x)       

def LogisticRegressionWeightedQuadratic(DTR,LTR,DTE,LTE,lambd):

    def vecxxT(x):
        x = x[:,None]
        xxT = x.dot(x.T).reshape(x.size **2, order='F')
        return xxT
    
    trasformed_DTR = numpy.apply_along_axis(vecxxT,0,DTR)
    trasformed_DTE = numpy.apply_along_axis(vecxxT, 0 ,DTE)
    phi_DTR = numpy.array(numpy.vstack([trasformed_DTR,DTR]))

    phi_DTE = numpy.array(numpy.vstack([trasformed_DTE,DTE]))

    # ----- WITH GRADIENT, SO WITHOUT APPROX_GRAD=TRUE  ------
    logreg_obj_quadratic_weighted_gradient = logreg_obj_wrap_weighted_gradient(phi_DTR, LTR, lambd) 
    (x, f, d) = scipy.optimize.fmin_l_bfgs_b(logreg_obj_quadratic_weighted_gradient, numpy.zeros(phi_DTR.shape[0] + 1))
    # ----- WITHOUT GRADIENT CALL   -------
    #(x, f, d) = scipy.optimize.fmin_l_bfgs_b(logreg_obj_quadratic_weighted, numpy.zeros(phi_DTR.shape[0] + 1), approx_grad=True)
    #print("Lambda=" + str(lambd)) 
    #print(f"The objective value at the minimum (J(w*,b*)) is: {round(f, 7)}") 
    return computeScores(phi_DTE, LTE, x) 

def logreg_obj_wrap_weighted_gradient(DTR, LTR, lambd):
    def logreg_obj_weighted_gradient(v):
        w, b = v[0:-1], v[-1]
        n = DTR.shape[1]
        first_term = (lambd / 2) * (numpy.linalg.norm(w) ** 2)
        DP0, DP1 = main.getClassMatrix(DTR, LTR)
        app_prior = constants.EFFECTIVE_PRIOR
        weight_0 = (1 - app_prior) / DP0.shape[1]
        weight_1 = app_prior / DP1.shape[1]
        loss_0 = 0
        loss_1 = 0
        G1 = lambd * w
        G2 = 0

        for i in range(n):
            c_i = LTR[i]
            z_i = 2 * c_i - 1
            x_i = DTR[:, i]
            term = numpy.exp(-z_i * ((numpy.dot(w.T, x_i)) + b))
            loss_derivative = term / (1 + term)

            if z_i == -1:
                # class 0
                loss_0 += numpy.logaddexp(0, -z_i * ((numpy.dot(w.T, x_i)) + b))
                G1 += weight_0 * (-z_i * x_i) * loss_derivative
                G2 += weight_0 * (-z_i) * loss_derivative
            else:
                # class 1
                loss_1 += numpy.logaddexp(0, -z_i * ((numpy.dot(w.T, x_i)) + b))
                G1 += weight_1 * (-z_i * x_i) * loss_derivative
                G2 += weight_1 * (-z_i) * loss_derivative

        function = first_term + weight_1 * loss_1 + weight_0 * loss_0
        gradient = numpy.concatenate((G1, [G2]))

        return function, gradient

    return logreg_obj_weighted_gradient