import numpy
import scipy 
import main
import constants

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

def logreg_obj_wrap_weighted(DTR, LTR, lambd):
    def logreg_obj_weighted(v):
        w, b = v[0:-1], v[-1]
        n = DTR.shape[1]
        first_term = (lambd/2) * (numpy.linalg.norm(w)**2)
        DP0,DP1 = main.getClassMatrix(DTR,LTR)
        app_prior = constants.PRIOR_PROBABILITY
        # riga 30 e 33 come Laura la prior
        # (non ha senso è gia data così i 2 weight sono uguali, come se facessi il non weighted con 1/n dato che dai stesso peso infatti viene uguale)
        #app_prior = DP1.shape[1]/DTR.shape[1]
        weight_0 = (1-app_prior)/DP0.shape[1]
        weight_1 = app_prior/DP1.shape[1]
        #print(weight_0 == weight_1)
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

def computeScores_quad(DTR, LTR, DTE, LTE):
    def vec(A):
        # Get the number of rows and columns in A
        m, n = A.shape

        # Reshape the transpose of A into a column vector
        return numpy.reshape(numpy.transpose(A), (m * n, 1))

    DP0,DP1 = main.getClassMatrix(DTR,LTR)
    mu0,C0 = main.computeMeanCovMatrix(DP0)
    mu1,C1 = main.computeMeanCovMatrix(DP1)
    l0 = (numpy.linalg.inv(C0))
    l1 = (numpy.linalg.inv(C1))
    A = -0.5 * (l1-l0)
    b = numpy.dot(l1,mu1) - numpy.dot(l0,mu0)
    c = -0.5 * numpy.dot( mu1.T, numpy.dot(l1, mu1)) - numpy.dot( mu0.T, numpy.dot(l0, mu0)) + 0.5*(numpy.linalg.slogdet(l1)[1]-numpy.linalg.slogdet(l0)[1])
    #w = [vec(A),b.reshape((10,1))]
    w_t = [vec(A).T,b.reshape((10,1)).T]
    LP = []
    for i in range(0,DTE.shape[1]):
        x = DTE[:,i]
        x = x.reshape((10,1))
        res = vec(numpy.dot(x,x.T))
        phi = [res , x]
        s_i = numpy.dot(w_t[0],phi[0]) + numpy.dot(w_t[1],phi[1]) + c
        if s_i > 0:
            LP.append(1)
        else:
            LP.append(0)
    nCorrectPredictions = computeNumCorrectPredictionsDiscriminative(numpy.array(LP),LTE)
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

def LogisticRegressionWeighted(DTR,LTR,DTE,LTE):
    lambd = 0.1
    #lambda_values = [0.000001, 0.001, 0.1, 1.0] 
    #for lambd in lambda_values: 
    logreg_obj_weighted = logreg_obj_wrap_weighted(DTR, LTR, lambd) 
    (x, f, d) = scipy.optimize.fmin_l_bfgs_b(logreg_obj_weighted, numpy.zeros(DTR.shape[0] + 1), approx_grad=True)
    #print("Lambda=" + str(lambd)) 
    #print(f"The objective value at the minimum (J(w*,b*)) is: {round(f, 7)}") 
    return computeScores(DTE, LTE, x)       