import constants

class PlotUtility:
  def __init__(self, prior=constants.PRIOR_PROBABILITY, k=None, C=None, c=None, d=None, gamma=None, minDcf=None):
    self.prior = prior
    self.k = k
    self.C = C
    self.c = c
    self.d = d
    self.gamma = gamma
    self.minDcf = minDcf
    
  def __str__(self):
    return f"Hyperparameters values: K:{self.k}\tC:{self.C}\tc:{self.c}\td:{self.d}\tgamma:{self.gamma}\nwith prior:{self.prior}\nfor minDcf value:{self.minDcf}"
    
  def is_prior(self, prior):
    return self.prior == prior

  def is_k(self, k):
    return self.k == k
    
  def is_C(self, C):
    return self.C == C

  def is_c(self, c):
    return self.c == c
 
  def is_d(self, d):
    return self.d == d

  def is_gamma(self, gamma):
    return self.gamma == gamma
  
  def getminDcf(self):
    return self.minDcf
  
  def getC(self):
    return self.C

# -- USAGE EXAMPLE ---

# x1 = PlotUtility(C=2,c=3,d=4,gamma=5,minDcf=0.5)
# x2 = PlotUtility(k=2,C=3,c=4,d=5,gamma=6,minDcf=0.5)
# x3 = PlotUtility(k=2,C=4,c=5,d=6,gamma=7,minDcf=0.5)
# x4 = PlotUtility(k=4,C=5,c=6,d=7,gamma=8,minDcf=0.5)

# l = [x1,x2,x3,x4]

# # filter for plot only desider values
# filtered = filter(lambda PlotElement: PlotElement.is_k(2) and PlotElement.is_C(3), l)

# # get then only mindcf values for plotting
# #minDcfs = []
# # remaining elements
# #for el in filtered:
# #    print(el)
# #    print()
# #    minDcfs.append(el.getminDcf())


# # get then only mindcf values for plotting
# minDcfs = [PlotElement.getminDcf() for PlotElement in filtered]

# print(minDcfs)