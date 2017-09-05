import matplotlib.pyplot as plt
import numpy as np
#from scipy.optimize import fsolve
sqrt = np.emath.sqrt
from scipy.optimize import fsolve
from scipy.optimize import root
np.random.seed(0)
d=16
x0=np.random.randint(1,size=(d,d))
def mcmc_sampling(J,x=[[]]):
    mt0,l=300,len(x[0])
    tst=0
    for t in range(mt0):
        for i in range(l):
            for j in range(l):
                valu=( ( x[(i+l-1)%l][j] + x[(i+1)%l][j] + x[i][(j+l-1)%l] + x[i][(j+1)%l] ) )
                r=np.exp(-J*valu)/(np.exp(J*valu) + np.exp(-J*valu))
                R=np.random.uniform(0,1)
                if(R<=r):
                    x[i][j]=1
                else:
                    x[i][j]=-1

    return x

##############main##############
print(x0)
plt.subplot(131)
plt.imshow(x0,interpolation='none')

x=mcmc_sampling(1/d,x0)
print(x)
plt.subplot(132)
plt.imshow(x,interpolation='none')

x1=np.random.randint(1,size=(d,d))
x1=mcmc_sampling(0/d,x1)
print(x1)
plt.subplot(133)
plt.imshow(x1,interpolation='none')
plt.show()
