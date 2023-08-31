import numpy 
import scipy
import matplotlib.pyplot as plt
import main 
import generative_models

#INIZIO LAB10
def logpdf_GMM(X, gmm): #Function that computes the log-density of a GMM [log(fXi(xi)) = SOMMATORIA CHE VA DA g=1 A N DI log fXi,Gi(xi,g)] for a set of samples xi contained in matrix X.
    # You can arrange the terms in a matrix S with shape M, N. 
    # Each row S[g, :] of S contains the (sub-)class conditional densities given component Gi = g for all samples xi. 
    S = numpy.zeros((len(gmm), X.shape[1])) # shape of 2 x 2325 (or less of 2325 if k-fold is applied) 
    for g in range(len(gmm)):
        # You can compute the joint log-density log fXi,Gi(xi,g) for each component g of the GMM and each sample by 
        # adding, to each row of S, the logarithm of the prior of the corresponding component log P(Gi = g)=log wg:
        S[g, :] = generative_models.logpdf_GAU_ND(X, gmm[g][1], gmm[g][2]) #PASSIAMO mug, Cg --> #WE can make use of the function generative_models.logpdf_GAU_ND that you wrote for Laboratory 4 to compute the terms log fXi|Gi(xi|g) = log N(xi|µg,Σg) for all samples xi, i = 1...N and for all components g = 1...M. 
        S[g, :] += numpy.log(gmm[g][0]) #fXi,Gi(xi,g) = fXi|Gi(xi|g)*P(Gi = g) --> log fXi,Gi(xi,g) = log fXi|Gi(xi|g)+log P(Gi = g)
    logdens = scipy.special.logsumexp(S, axis=0) #log-marginal log(fXi(xi)) --> #The result will be an array of shape (N,), whose component i will contain the log-density for sample xi #BASICALLY IT'S LIKE WE ARE DOING: np.log(np.sum(np.exp(a))) --> COSì FACCIAMO IL LOGARITMO DELLA SOMMA!
    #print(logdens.shape)
    return (logdens, S)
# N.B.: The process is the same we used to compute the log-densities for the MVG classifier in Laboratory 5. 
# Indeed, we can interpret Gaussian components G as representing (sub)classes of our data.

# E-step: compute the POSTERIOR PROBABILITY γg,i = P(Gi = g|Xi = xi, Mt, St, wt) --> vedi FINE PAGINA 2 X FORMULA
# for each component of the GMM for each sample, using an estimate (Mt,St, wt) of the model parameters. 
# These quantities are also called responsibilities.
#The computation of the posterior distributions γg,i can be done in the same way as we computed class posterior 
# probabilities for a MVG model. We can also re-use most of the code that was used to compute in the previous section.
def Estep(logdens, S):
    # You can compute the logarithm of all γg,i's by removing, from each row of matrix S (WHICH CONTAINS 
    # joint densities), the row vector containing the N marginal densities computed in the previous step. 
    # The M × N matrix of posterior probabilities can then be obtained by computing 
    # the exponential of each element of the result.
    posterior_probabilities = numpy.exp(S-logdens.reshape(1, logdens.size))
    return posterior_probabilities

#NOW WE UPDATE THE MODEL PARAMETERS.
# We can use the statistics Zg, Fg, Sg to obtain the new parameters µgt+1, Σgt+1, wgt+1, g = 1...M. 
def Mstep(X, S, posterior_probabilities):
    #LET'S COMPUTE THE 3 STATISTICS:
    Zg = numpy.sum(posterior_probabilities, axis=1)  #3
    #print(Zg)
    Fg = numpy.zeros((X.shape[0], S.shape[0]))  # 4x3
    for g in range(S.shape[0]):
        tempSum = numpy.zeros(X.shape[0])
        for i in range(X.shape[1]):
            tempSum += posterior_probabilities[g, i] * X[:, i]
        Fg[:, g] = tempSum
    #print(Fg)
    Sg = numpy.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        tempSum = numpy.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[1]):
            tempSum += posterior_probabilities[g, i] * numpy.dot(X[:, i].reshape((X.shape[0], 1)), X[:, i].reshape((1, X.shape[0])))
        Sg[g] = tempSum
    #print(Sg)
    #print(Sg.shape) #(3,4,4) OR (3,1,1), IT DEPENDS ON WHAT DATA WE ARE USING: 4-DIMENSIONAL OR 1-DIMENSIONAL DATA
    #LET'S obtain the new parameters:
    mu = Fg / Zg
    #print(mu)
    prodmu = numpy.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        prodmu[g] = numpy.dot(mu[:, g].reshape((X.shape[0], 1)), mu[:, g].reshape((1, X.shape[0])))
    cov = Sg / Zg.reshape((Zg.size, 1, 1)) - prodmu
    # The following two lines of code are used to model the constraints
    psi = 0.01 #ψ
    for g in range(S.shape[0]):
        # To avoid degenerate solutions, we can constrain the minimum
        # values of the eigenvalues of the covariance matrices.
        # A possible solution consists in constraining the eigenvalues of the 
        # covariance matrices to be larger or equal to a lower bound ψ > 0
        U, s, Vh = numpy.linalg.svd(cov[g])
        s[s < psi] = psi
        cov[g] = numpy.dot(U, main.vcol(s)*U.T)
    # print(cov)
    w = Zg / numpy.sum(Zg)
    # print(w)
    return (w, mu, cov)

#The EM algorithm can be used to estimate the parameters of a GMM that maximize the likelihood for
#a training set X. The EM algorithm consists of two steps: 1)E-step and 2)M-step.
def EMalgorithm(X, gmm):
    #The estimation procedure should iterate from an initial
    #estimate of the GMM, and keep computing E and M -steps until a convergence criterion is met.
    # flag is used to exit the while when the difference between the loglikelihoods
    # becomes smaller than THE THRESHOLD delta_l = 10^(-6)
    flag = True
    # count is used to count iterations
    count = 0
    while(flag):
        count += 1
        # Given the training set and the initial model parameters, compute log marginal densities and sub-class conditional densities
        (logdens, S) = logpdf_GMM(X, gmm) #The log-likelihood for the training set using the current parameter estimate can be computed from the outputs of function logpdf_GMM. Alternatively, you can compute the log-likelihood from the marginal densities at the end of the E-step
        # Compute the AVERAGE loglikelihood, by summing all the log densities and dividing by the number of samples (it's as if we're computing a mean)
        loglikehood1 = numpy.sum(logdens)
        avgloglikelihood1 = loglikehood1/X.shape[1] # In the following we will use the average log-likelihood, i.e. we will divide the loglikelihood by N.
        posterior_probabilities = Estep(logdens, S)
        (w, mu, cov) = Mstep(X, S, posterior_probabilities)
        for g in range(len(gmm)):
            # Update the model parameters that are in gmm
            U,s,_=numpy.linalg.svd(cov[g])
            s[s<0.01] = 0.01
            cov[g]= numpy.dot(U, main.vcol(s)*U.T)
            gmm[g] = (w[g], mu[:, g].reshape((mu.shape[0], 1)), cov[g])
        # Compute the new log densities and the new sub-class conditional densities
        (logdens, S) = logpdf_GMM(X, gmm)
        loglikehood2 = numpy.sum(logdens)
        avgloglikelihood2 =  loglikehood2/X.shape[1] #PDF: The average log-likelihood for the trained GMM is -7.26325603 (WHEN WORKING WITH 4-DIMENSIONAL DATA)
        #if(X.shape[0] == 4):
        #    print("Average log-likelihood 4D-DATA: ", avgloglikelihood2) #CI INTERESSA IL RISULTATO DI QUESTA COSì DA POTERLA CONFRONTARE CON IL PDF --> AD OGNI MODO BASTA STAMPARE L'ULTIMA CHE OTTENIAMO, ANZICHé FARE UNA PRINT AD OGNI ITERAZIONE
        #if(X.shape[0] == 1):
            #print("Average log-likelihood 1D-DATA: ", avgloglikelihood2)
        if (avgloglikelihood2-avgloglikelihood1 < 10**(-6)): #We stop the iterations when the average log-likelihood increases by a value lower than a threshold
            flag = False
        if (avgloglikelihood2-avgloglikelihood1 < 0): #To verify your implementation, check your log-likelihood is increasing. If the loglikelihood becomes smaller at some iteration, then your implementation is very likely to be incorrect!
            print("The loglikelihood has become smaller at iteration: " + str(count))
    #print(count) 
    # if(X.shape[0] == 4):
    #     print("Average log-likelihood 4D-DATA: ", avgloglikelihood2)
    #if(X.shape[0] == 1):
        #print("Average log-likelihood 1D-DATA: ", avgloglikelihood2)
    return gmm 

'''
def plotEstimatedDensity_and_hystogram(dataset, gmm, colore):
    # Function used to plot the computed normal density over the normalized histogram
    plt.figure()
    plt.hist(dataset, bins=30, edgecolor='purple', linewidth=0.5, density=True)
    # Define an array of equidistant 1000 elements between -10 and 5
    XPlot = numpy.linspace(-10, 5, 1000)
    # We should plot the density, not the log-density, so we need to use numpy.exp
    y = numpy.zeros(1000)
    for g in range(len(gmm)):
        y += gmm[g][0]*numpy.exp(logpdf_GAU_1D(XPlot, gmm[g][1], gmm[g][2])).flatten() #NON SO SE LA FLATTEN è NECESSARIA
    plt.plot(XPlot, y, color=colore, linewidth=3)
'''

#SPLIT a GMM with G component(s) TO OBTAIN a GMM with 2G components
def split(GMM):
    alpha = 0.1
    size = len(GMM)
    splittedGMM = []
    for i in range(size):
        # The displacement vector dg can be computed, for example, by taking the leading eigenvector of Σg,
        # scaled by the square root of the corresponding eigenvalue, multiplied by some factor α
        U, s, Vh = numpy.linalg.svd(GMM[i][2])
        dg = U[:, 0:1] * s[0] ** 0.5 * alpha
        #A possible way to split the GMM consists in replacing component (wg, µg, Σg) with two components:
        #(wg/2, µg+dg, Σg), (wg/2, µg-dg, Σg)
        splittedGMM.append((GMM[i][0]/2, GMM[i][1]+dg, GMM[i][2]))
        splittedGMM.append((GMM[i][0]/2, GMM[i][1]-dg, GMM[i][2]))
        # In practice, we are displacing the new components along the direction of maximum variance, using a
        # step that is proportional to the standard deviation (i.e. the scale of the features along that 
        # direction) of the component we are splitting.
    # print("Splitted GMM", splittedGMM)
    return splittedGMM

# The EM algorithm requires an initial guess for the GMM parameters. The LBG algorithm allows us
# to incrementally construct a GMM with 2G components from a GMM with G components. Starting
# with a single-component GMM (i.e. a Gaussian density), we can build a 2-component GMM and then
# use the EM algorithm to estimate a ML solution for the 2-components model. We can then split the 2
# components to obtain a 4-components GMM, an re-apply the EM algorithm to estimate its parameters, and so on.
def LBGalgorithm(GMM, X, iterations):
    GMM = EMalgorithm(X, GMM) #AT THE BEGINNING IT IS A 1G-COMPONENT GMM
    # At each LBG iteration, we produce a 2G-components GMM from a G-components GMM. The 2G components GMM can 
    # be used as initial GMM for the EM algorithm --> WE Retrain the 2G-GMM using the EM algorithm:
    for i in range(iterations):
        GMM_2G_COMPONENTS = split(GMM) #GMM HERE IN GENERAL IS A G-components GMM (WE MAY START WITH A SINGLE COMPONENT WITH G=1, BUT IN THE NEXT ITERATIONS WE WILL HAVE MORE THAN ONE COMPONENT)
        GMM = EMalgorithm(X, GMM_2G_COMPONENTS) 
        #N.B.: After the split, the initial iterations of the EM algorithm for the 2G-components GMM may
        #have a lower log-likelihood than you had at the end of the EM iterations for the G-components GMM.
    return GMM

def DiagMstep(X, S, posterior_probabilities):
    #LET'S COMPUTE THE 3 STATISTICS:
    Zg = numpy.sum(posterior_probabilities, axis=1)  #3
    #print(Zg)
    Fg = numpy.zeros((X.shape[0], S.shape[0]))  # 4x3
    for g in range(S.shape[0]):
        tempSum = numpy.zeros(X.shape[0])
        for i in range(X.shape[1]):
            tempSum += posterior_probabilities[g, i] * X[:, i]
        Fg[:, g] = tempSum
    #print(Fg)
    Sg = numpy.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        tempSum = numpy.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[1]):
            tempSum += posterior_probabilities[g, i] * numpy.dot(X[:, i].reshape((X.shape[0], 1)), X[:, i].reshape((1, X.shape[0])))
        Sg[g] = tempSum
    #print(Sg)
    #print(Sg.shape) #(3,4,4) OR (3,1,1), IT DEPENDS ON WHAT DATA WE ARE USING: 4-DIMENSIONAL OR 1-DIMENSIONAL DATA
    #LET'S obtain the new parameters:
    mu = Fg / Zg
    #print(mu)
    prodmu = numpy.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        prodmu[g] = numpy.dot(mu[:, g].reshape((X.shape[0], 1)), mu[:, g].reshape((1, X.shape[0])))
    cov = Sg / Zg.reshape((Zg.size, 1, 1)) - prodmu
    # The following two lines of code are used to model the constraints
    psi = 0.01
    for g in range(S.shape[0]):
        cov[g] = cov[g] * numpy.eye(cov[g].shape[0]) #CAMBIA SOLO CHE AGGIUNGIAMO QUESTA RIGA RISPETTO A MStep NORMALE
        U, s, Vh = numpy.linalg.svd(cov[g])
        s[s < psi] = psi
        cov[g] = numpy.dot(U, main.vcol(s)*U.T)
    # print(cov)
    w = Zg / numpy.sum(Zg)
    # print(w)
    return (w, mu, cov)

def DiagEMalgorithm(X, gmm): 
    #The estimation procedure should iterate from an initial
    #estimate of the GMM, and keep computing E and M -steps until a convergence criterion is met.
    # flag is used to exit the while when the difference between the loglikelihoods
    # becomes smaller than THE THRESHOLD delta_l = 10^(-6)
    flag = True
    # count is used to count iterations
    count = 0
    while(flag):
        count += 1
        # Given the training set and the initial model parameters, compute log marginal densities and sub-class conditional densities
        (logdens, S) = logpdf_GMM(X, gmm) #The log-likelihood for the training set using the current parameter estimate can be computed from the outputs of function logpdf_GMM. Alternatively, you can compute the log-likelihood from the marginal densities at the end of the E-step
        # Compute the AVERAGE loglikelihood, by summing all the log densities and dividing by the number of samples (it's as if we're computing a mean)
        loglikehood1 = numpy.sum(logdens)
        avgloglikelihood1 = loglikehood1/X.shape[1] # In the following we will use the average log-likelihood, i.e. we will divide the loglikelihood by N.
        posterior_probabilities = Estep(logdens, S)
        (w, mu, cov) = DiagMstep(X, S, posterior_probabilities) #CAMBIA SOLO LA FUNZIONE CHE INVOCHIAMO QUI RISPETTO A EMalgorithm NORMALE
        for g in range(len(gmm)):
            # Update the model parameters that are in gmm
            U,s,_=numpy.linalg.svd(cov[g])
            s[s<0.01] = 0.01
            cov[g]= numpy.dot(U, main.vcol(s)*U.T)
            gmm[g] = (w[g], mu[:, g].reshape((mu.shape[0], 1)), cov[g])
        # Compute the new log densities and the new sub-class conditional densities
        (logdens, S) = logpdf_GMM(X, gmm)
        loglikehood2 = numpy.sum(logdens)
        avgloglikelihood2 =  loglikehood2/X.shape[1] #PDF: The average log-likelihood for the trained GMM is -7.26325603 (WHEN WORKING WITH 4-DIMENSIONAL DATA)
        if (avgloglikelihood2-avgloglikelihood1 < 10**(-6)): #We stop the iterations when the average log-likelihood increases by a value lower than a threshold
            flag = False
        if (avgloglikelihood2-avgloglikelihood1 < 0): #To verify your implementation, check your log-likelihood is increasing. If the loglikelihood becomes smaller at some iteration, then your implementation is very likely to be incorrect!
            print("The loglikelihood has become smaller at iteration: " + str(count))
    #print(count) 
    # if(X.shape[0] == 4):
    #     print("Average log-likelihood 4D-DATA: ", avgloglikelihood2)
    #if(X.shape[0] == 1):
        #print("Average log-likelihood 1D-DATA: ", avgloglikelihood2)
    return gmm 

# The ML solution can be trivially obtained by modifying the M-step by
# keeping only the diagonal elements of Σgt+1
def DiagLBGalgorithm(GMM, X, iterations):
    GMM = DiagEMalgorithm(X, GMM)
    for i in range(iterations):
        GMM = split(GMM)
        GMM = DiagEMalgorithm(X, GMM)
    return GMM

def TiedMstep(X, S, posterior_probabilities):
    #LET'S COMPUTE THE 3 STATISTICS:
    Zg = numpy.sum(posterior_probabilities, axis=1)  #3
    #print(Zg)
    Fg = numpy.zeros((X.shape[0], S.shape[0]))  # 4x3
    for g in range(S.shape[0]):
        tempSum = numpy.zeros(X.shape[0])
        for i in range(X.shape[1]):
            tempSum += posterior_probabilities[g, i] * X[:, i]
        Fg[:, g] = tempSum
    #print(Fg)
    Sg = numpy.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        tempSum = numpy.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[1]):
            tempSum += posterior_probabilities[g, i] * numpy.dot(X[:, i].reshape((X.shape[0], 1)), X[:, i].reshape((1, X.shape[0])))
        Sg[g] = tempSum
    #print(Sg)
    #print(Sg.shape) #(3,4,4) OR (3,1,1), IT DEPENDS ON WHAT DATA WE ARE USING: 4-DIMENSIONAL OR 1-DIMENSIONAL DATA
    #LET'S obtain the new parameters:
    mu = Fg / Zg
    #print(mu)
    prodmu = numpy.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        prodmu[g] = numpy.dot(mu[:, g].reshape((X.shape[0], 1)), mu[:, g].reshape((1, X.shape[0])))
    cov = Sg / Zg.reshape((Zg.size, 1, 1)) - prodmu
    # The following two lines of code are used to model the constraints
    psi = 0.01
    term = numpy.zeros((cov.shape[1], cov.shape[2])) #cambiamo solo questa e le 4 righe successive rispetto a Mstep
    for g in range(S.shape[0]):
        term += Zg[g]*cov[g]
    for g in range(S.shape[0]):
        cov[g] = 1/X.shape[1] * term
        U, s, Vh = numpy.linalg.svd(cov[g])
        s[s < psi] = psi
        cov[g] = numpy.dot(U, main.vcol(s)*U.T)
    # print(cov)
    w = Zg / numpy.sum(Zg)
    # print(w)
    return (w, mu, cov)

def TiedDiagMstep(X, S, posterior_probabilities):
    #LET'S COMPUTE THE 3 STATISTICS:
    Zg = numpy.sum(posterior_probabilities, axis=1)  #3
    #print(Zg)
    Fg = numpy.zeros((X.shape[0], S.shape[0]))  # 4x3
    for g in range(S.shape[0]):
        tempSum = numpy.zeros(X.shape[0])
        for i in range(X.shape[1]):
            tempSum += posterior_probabilities[g, i] * X[:, i]
        Fg[:, g] = tempSum
    #print(Fg)
    Sg = numpy.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        tempSum = numpy.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[1]):
            tempSum += posterior_probabilities[g, i] * numpy.dot(X[:, i].reshape((X.shape[0], 1)), X[:, i].reshape((1, X.shape[0])))
        Sg[g] = tempSum
    #print(Sg)
    #print(Sg.shape) #(3,4,4) OR (3,1,1), IT DEPENDS ON WHAT DATA WE ARE USING: 4-DIMENSIONAL OR 1-DIMENSIONAL DATA
    #LET'S obtain the new parameters:
    mu = Fg / Zg
    #print(mu)
    prodmu = numpy.zeros((S.shape[0], X.shape[0], X.shape[0]))
    for g in range(S.shape[0]):
        prodmu[g] = numpy.dot(mu[:, g].reshape((X.shape[0], 1)), mu[:, g].reshape((1, X.shape[0])))
    cov = Sg / Zg.reshape((Zg.size, 1, 1)) - prodmu
    # The following two lines of code are used to model the constraints
    psi = 0.01
    term = numpy.zeros((cov.shape[1], cov.shape[2])) #cambiamo solo questa e le 4 righe successive rispetto a Mstep
    for g in range(S.shape[0]):
        term += Zg[g]*cov[g]
    for g in range(S.shape[0]):
        # TIED + DIAGONAL
        cov[g] = 1/X.shape[1] * term
        cov[g] = cov[g] * numpy.eye(cov[g].shape[0]) #CAMBIA SOLO CHE AGGIUNGIAMO QUESTA RIGA RISPETTO A MStep NORMALE
        U, s, Vh = numpy.linalg.svd(cov[g])
        s[s < psi] = psi
        cov[g] = numpy.dot(U, main.vcol(s)*U.T)
    # print(cov)
    w = Zg / numpy.sum(Zg)
    # print(w)
    return (w, mu, cov)

#each component has a covariance matrix Σg = Σ
def TiedEMalgorithm(X, gmm): #
    #The estimation procedure should iterate from an initial
    #estimate of the GMM, and keep computing E and M -steps until a convergence criterion is met.
    # flag is used to exit the while when the difference between the loglikelihoods
    # becomes smaller than THE THRESHOLD delta_l = 10^(-6)
    flag = True
    # count is used to count iterations
    count = 0
    while(flag):
        count += 1
        # Given the training set and the initial model parameters, compute log marginal densities and sub-class conditional densities
        (logdens, S) = logpdf_GMM(X, gmm) #The log-likelihood for the training set using the current parameter estimate can be computed from the outputs of function logpdf_GMM. Alternatively, you can compute the log-likelihood from the marginal densities at the end of the E-step
        # Compute the AVERAGE loglikelihood, by summing all the log densities and dividing by the number of samples (it's as if we're computing a mean)
        loglikehood1 = numpy.sum(logdens)
        avgloglikelihood1 = loglikehood1/X.shape[1] # In the following we will use the average log-likelihood, i.e. we will divide the loglikelihood by N.
        posterior_probabilities = Estep(logdens, S)
        (w, mu, cov) = TiedMstep(X, S, posterior_probabilities) #CAMBIA SOLO LA FUNZIONE CHE INVOCHIAMO QUI RISPETTO A EMalgorithm NORMALE
        for g in range(len(gmm)):
            # Update the model parameters that are in gmm
            U,s,_=numpy.linalg.svd(cov[g])
            s[s<0.01] = 0.01
            cov[g]= numpy.dot(U, main.vcol(s)*U.T)
            gmm[g] = (w[g], mu[:, g].reshape((mu.shape[0], 1)), cov[g])
        # Compute the new log densities and the new sub-class conditional densities
        (logdens, S) = logpdf_GMM(X, gmm)
        loglikehood2 = numpy.sum(logdens)
        avgloglikelihood2 =  loglikehood2/X.shape[1] #PDF: The average log-likelihood for the trained GMM is -7.26325603 (WHEN WORKING WITH 4-DIMENSIONAL DATA)
        if (avgloglikelihood2-avgloglikelihood1 < 10**(-6)): #We stop the iterations when the average log-likelihood increases by a value lower than a threshold
            flag = False
        if (avgloglikelihood2-avgloglikelihood1 < 0): #To verify your implementation, check your log-likelihood is increasing. If the loglikelihood becomes smaller at some iteration, then your implementation is very likely to be incorrect!
            print("The loglikelihood has become smaller at iteration: " + str(count))
    #print(count) 
    # if(X.shape[0] == 4):
    #     print("Average log-likelihood 4D-DATA: ", avgloglikelihood2)
    #if(X.shape[0] == 1):
        #print("Average log-likelihood 1D-DATA: ", avgloglikelihood2)
    return gmm 

def TiedDiagEMalgorithm(X, gmm):
    #The estimation procedure should iterate from an initial
    #estimate of the GMM, and keep computing E and M -steps until a convergence criterion is met.
    # flag is used to exit the while when the difference between the loglikelihoods
    # becomes smaller than THE THRESHOLD delta_l = 10^(-6)
    flag = True
    # count is used to count iterations
    count = 0
    while(flag):
        count += 1
        # Given the training set and the initial model parameters, compute log marginal densities and sub-class conditional densities
        (logdens, S) = logpdf_GMM(X, gmm) #The log-likelihood for the training set using the current parameter estimate can be computed from the outputs of function logpdf_GMM. Alternatively, you can compute the log-likelihood from the marginal densities at the end of the E-step
        # Compute the AVERAGE loglikelihood, by summing all the log densities and dividing by the number of samples (it's as if we're computing a mean)
        loglikehood1 = numpy.sum(logdens)
        avgloglikelihood1 = loglikehood1/X.shape[1] # In the following we will use the average log-likelihood, i.e. we will divide the loglikelihood by N.
        posterior_probabilities = Estep(logdens, S)
        (w, mu, cov) = TiedDiagMstep(X, S, posterior_probabilities) #CAMBIA SOLO LA FUNZIONE CHE INVOCHIAMO QUI RISPETTO A EMalgorithm NORMALE
        for g in range(len(gmm)):
            # Update the model parameters that are in gmm
            U,s,_=numpy.linalg.svd(cov[g])
            s[s<0.01] = 0.01
            cov[g]= numpy.dot(U, main.vcol(s)*U.T)
            gmm[g] = (w[g], mu[:, g].reshape((mu.shape[0], 1)), cov[g])
        # Compute the new log densities and the new sub-class conditional densities
        (logdens, S) = logpdf_GMM(X, gmm)
        loglikehood2 = numpy.sum(logdens)
        avgloglikelihood2 =  loglikehood2/X.shape[1] #PDF: The average log-likelihood for the trained GMM is -7.26325603 (WHEN WORKING WITH 4-DIMENSIONAL DATA)
        if (avgloglikelihood2-avgloglikelihood1 < 10**(-6)): #We stop the iterations when the average log-likelihood increases by a value lower than a threshold
            flag = False
        if (avgloglikelihood2-avgloglikelihood1 < 0): #To verify your implementation, check your log-likelihood is increasing. If the loglikelihood becomes smaller at some iteration, then your implementation is very likely to be incorrect!
            print("The loglikelihood has become smaller at iteration: " + str(count))
    #print(count) 
    # if(X.shape[0] == 4):
    #     print("Average log-likelihood 4D-DATA: ", avgloglikelihood2)
    #if(X.shape[0] == 1):
        #print("Average log-likelihood 1D-DATA: ", avgloglikelihood2)
    return gmm 

def TiedLBGalgorithm(GMM, X, iterations):
    GMM = TiedEMalgorithm(X, GMM)
    for i in range(iterations):
        GMM = split(GMM)
        GMM = TiedEMalgorithm(X, GMM)
    return GMM

def TiedDiagLBGalgorithm(GMM, X, iterations):
    GMM = TiedDiagEMalgorithm(X, GMM)
    for i in range(iterations):
        GMM = split(GMM)
        GMM = TiedDiagEMalgorithm(X, GMM)
    return GMM

def constraintSigma(sigma):
    psi = 0.01
    U, s, Vh = numpy.linalg.svd(sigma)
    s[s < psi] = psi
    sigma = numpy.dot(U, main.vcol(s)*U.T)
    return sigma

def DiagConstraintSigma(sigma):
    # same for diag and tied diag
    sigma = sigma * numpy.eye(sigma.shape[0])
    psi = 0.01
    U, s, Vh = numpy.linalg.svd(sigma)
    s[s < psi] = psi
    sigma = numpy.dot(U, main.vcol(s)*U.T)
    return sigma

def GMM_Classifier(DTR0, DTR1, DTE, LTE, algorithm, n_split_0, n_split_1=None, constraint=None):
    D = [DTR0, DTR1] # Define a list that includes the three splitted training set
    marginalLikelihoods = [] # Define a list to store marginal likelihoods for the three sets
    # Iterate on the three sets
    for i in range(len(D)):
        wg = 1.0
        # Find mean and covariance matrices, reshape them as matrices because they
        # are scalar and in the following we need them as matrices
        mug = D[i].mean(axis=1).reshape((D[i].shape[0], 1))
        sigmag = constraint(numpy.cov(D[i]).reshape((D[i].shape[0], D[i].shape[0])))
        # Define initial component
        initialGMM = [(wg, mug, sigmag)]
        if n_split_1 == None:
            # same split for both classes (n_split_0)
            finalGMM = algorithm(initialGMM, D[i], n_split_0)
        elif i==0:
            # class 0, split is n_split_0
            finalGMM = algorithm(initialGMM, D[i], n_split_0)
        else:
            # class 1, split is n_split_1
            finalGMM = algorithm(initialGMM, D[i], n_split_1)
        # Compute marginal likelihoods and append them to the list
        marginalLikelihoods.append(logpdf_GMM(DTE, finalGMM)[0])
    # Stack all the likelihoods in PD
    PD = numpy.vstack((marginalLikelihoods[0], marginalLikelihoods[1]))
    # Compute the predicted labels
    predictedLabels = numpy.argmax(PD, axis=0)
    numberOfCorrectPredictions = numpy.array(predictedLabels == LTE).sum()
    accuracy = numberOfCorrectPredictions/LTE.size*100
    errorRate = 100-accuracy
    # marg_likelihood_class_1 - marg_likelihood_class_0 to compute the minDCF
    return marginalLikelihoods[1]-marginalLikelihoods[0] , numberOfCorrectPredictions