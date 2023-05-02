import matplotlib.pyplot as plt
import constants
import main 

def plot_hist(DTR,L):
    # 'spoofed-fingerprint' : name = 0 'authentic-fingerprint' : name = 1 
    spoofed_mask = (L == 0)
    authentic_mask = (L == 1)

    data_spoofed = DTR[:, spoofed_mask]
    data_authentic = DTR[:, authentic_mask]

    for i in range(0,constants.NUM_FEATURES):
        plt.figure(i)
        plt.xlabel("Feature" + str(i+1))
        plt.hist(data_spoofed[i, :], bins = 10, density = True, alpha = 0.4, label = 'Spoofed')
        plt.hist(data_authentic[i, :], bins = 10, density = True, alpha = 0.4, label = 'Authentic')
        plt.legend()
        plt.tight_layout()
    
    plt.show()

def plot_scatter(DTR,L):
    # 'spoofed-fingerprint' : name = 0 'authentic-fingerprint' : name = 1 
    spoofed_mask = (L == 0)
    authentic_mask = (L == 1)

    data_spoofed = DTR[:, spoofed_mask]
    data_authentic = DTR[:, authentic_mask]

    # plot only the unique combinations of the different features (we will have 45 only without dimensionality reduction)
    list_combination = []
    figure_id = 0
    for x in range(0,constants.NUM_FEATURES):
        for y in range(0,constants.NUM_FEATURES):
            if x == y:
                continue
            list_combination.append(str(y)+str(x))
            current_element = str(x)+str(y)
            if current_element in list_combination:
                continue
            figure_id+=1
            plt.figure(figure_id)
            plt.xlabel("Feature" + str(x+1))
            plt.ylabel("Feature" + str(y+1))
            plt.scatter(data_spoofed[x,:], data_spoofed[y,:], label = 'Spoofed')
            plt.scatter(data_authentic[x,:], data_authentic[y,:], label = 'Authentic')
            plt.legend()
            plt.tight_layout()
        plt.show()


def plot_scatter_projected_data_pca(DP,L):
    
    DP0,DP1 = main.getClassMatrix(DP,L)  
    
    # 2-D plot: regardless of the value of m, we can plot only for m = 2
    # for m=2 DPi[m-2: ], DPi[m-1 : ]
    plt.scatter(DP0[0, :], DP0[1, :], label = 'Spoofed')   
    plt.scatter(DP1[0, :], DP1[1, :], label = 'Authentic')  
 
    plt.legend()
    plt.show()

def plot_hist_projected_data_lda(DP,L):
    
    DP0,DP1 = main.getClassMatrix(DP,L)  
    
    # 1-D plot: 2 classes - 1 = 1
    plt.hist(DP0[0, :], bins = 10, density = True, alpha = 0.4, label = 'Spoofed')
    plt.hist(DP1[0, :], bins = 10, density = True, alpha = 0.4, label = 'Authentic')
    
    plt.legend()
    plt.show()