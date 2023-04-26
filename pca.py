import numpy
import main

def PCA_projection(DTR,m):
    
    _, C = main.computeMeanCovMatrix(DTR)

    s, U = numpy.linalg.eigh(C)

    P = U[:, ::-1][:, 0:m]
    
    DP = numpy.dot(P.T, DTR)
    
    #DP0,DP1 = getClassMatrix(DP,LTR)  
    
    # 2-D plot: regardless of the value of m, we can plot only for m = 2
    # for m=2 DPi[m-2: ], DPi[m-1 : ]
    #plt.scatter(DP0[0, :], DP0[1, :], label = 'Setosa')   
    #plt.scatter(DP1[0, :], DP1[1, :], label = 'Versicolor')  
    #plt.legend()
    #plt.show()

    # return the projected dataset
    return DP