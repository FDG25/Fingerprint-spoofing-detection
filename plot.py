import numpy
import scipy
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

# ------    PEARSON CORRELATION PLOTS   ----------

# pearson for the whole dataset

def plot_Heatmap_Whole_Dataset(DTR):
    heatmap = numpy.zeros((DTR.shape[0],DTR.shape[0]))
    for f1 in range(DTR.shape[0]):
        for f2 in range(DTR.shape[0]):
                if f2 <= f1:
                    heatmap[f1][f2] = abs(scipy.stats.pearsonr(DTR[f1, :], DTR[f2, :])[0])
                    heatmap[f2][f1] = heatmap[f1][f2]
    plt.figure() 
    plt.title('Pearson Correlation Coefficient of the Whole Dataset')
    plt.xticks(numpy.arange(0,constants.NUM_FEATURES),numpy.arange(1,constants.NUM_FEATURES + 1))  
    plt.yticks(numpy.arange(0,constants.NUM_FEATURES),numpy.arange(1,constants.NUM_FEATURES + 1))              
    plt.imshow(heatmap, cmap='Greys')
    plt.show()

# pearson for single class
# 'spoofed-fingerprint' : L = 0 (Red Color)
# 'authentic-fingerprint' : L = 1 (Blue Color)

def plot_Heatmap_Spoofed_Authentic(DTR, LTR, Class_Label):   
    heatmap = numpy.zeros((DTR.shape[0],DTR.shape[0]))
    for f1 in range(DTR.shape[0]):
        for f2 in range(DTR.shape[0]):
                if f2 <= f1:
                    heatmap[f1][f2] = abs(scipy.stats.pearsonr(DTR[:,LTR==Class_Label][f1, :], DTR[:,LTR==Class_Label][f2, :])[0])
                    heatmap[f2][f1] = heatmap[f1][f2]
    plt.figure()              
    plt.xticks(numpy.arange(0,constants.NUM_FEATURES),numpy.arange(1,constants.NUM_FEATURES + 1))  
    plt.yticks(numpy.arange(0,constants.NUM_FEATURES),numpy.arange(1,constants.NUM_FEATURES + 1))
    color = ''
    title = ''
    if Class_Label == 0:
        title = 'Pearson Correlation Coefficient of the spoofed-fingerprint class' 
        color = 'Reds'
    else:
        title = 'Pearson Correlation Coefficient of the authentic-fingerprint class' 
        color = 'Blues'
    plt.title(title)
    plt.imshow(heatmap, cmap=color )
    plt.show()

# -------   DCF PLOT    --------------
def plotDCF(x, y, xlabel):   
    plt.figure()
    plt.plot(x, y[0:len(x)], label='min DCF prior=0.5', color='b')    
    #plt.plot(x, y[len(x): 2*len(x)], label='min DCF prior=0.9', color='r')
    #plt.plot(x, y[2*len(x): 3*len(x)], label='min DCF prior=0.1', color='g')    
    plt.xlim([min(x), max(x)])
    plt.xscale("log")    
    #plt.legend(["min DCF prior=0.5", "min DCF prior=0.9", "min DCF prior=0.1"])
    plt.legend(["min DCF prior=0.5"])
    plt.xlabel(xlabel)    
    plt.ylabel("min DCF")
    plt.show()
    #plt.savefig('./images/dcf_lamba' + '.jpg')    
    return