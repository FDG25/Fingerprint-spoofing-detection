import numpy
import scipy 

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

def computeNumCorrectPredictionsDiscriminative(LP,LTE):
    bool_val = numpy.array(LP==LTE)
    n_samples = LTE.size
    acc = numpy.sum(bool_val)/n_samples
    err = 1 - acc
    return numpy.sum(bool_val)

def computeScores(DTE,LTE,v):
    w, b = v[0:-1], v[-1]
    n = DTE.shape[1]
    LP = []
    for i in range(0,n):
        x_t = DTE[:,i]
        s_i = numpy.dot(w.T,x_t)
        s_i+=b
        if s_i > 0:
            LP.append(1)
        else:
            LP.append(0)
    nCorrectPredictions = computeNumCorrectPredictionsDiscriminative(numpy.array(LP),LTE)
    #acc = nCorrectPredictions/LTE.size
    #err = 1 - acc
    #print("Accuracy: " + str(round(acc*100, 1)) + "%")
    #print("Error rate: " + str(round(err*100, 1)) + "%")
    return nCorrectPredictions

def LogisticRegression(DTR,LTR,DTE,LTE):
    lambd = 0.1
    #lambda_values = [0.000001, 0.001, 0.1, 1.0] 
    #for lambd in lambda_values: 
    logreg_obj = logreg_obj_wrap(DTR, LTR, lambd) 
    (x, f, d) = scipy.optimize.fmin_l_bfgs_b(logreg_obj, numpy.zeros(DTR.shape[0] + 1), approx_grad=True)
    #print("Lambda=" + str(lambd)) 
    #print(f"The objective value at the minimum (J(w*,b*)) is: {round(f, 7)}") 
    return computeScores(DTE, LTE, x) 
        