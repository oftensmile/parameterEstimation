#2016/05/19
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
np.random.seed(0)

#parameter 
d, N_sample = 124, 1000
theta=[[1 if i==(j+1+d)%d or i==(j-1+d)%d else 0 for i in range(d)] for j in range(d)]
X_sample = np.zeros((1,d))

def gen_mcmc(t_wait, x=[],theta=[[]]):
#Generate sample
for t in range(t_wait):
    for i in range(d):
        valu=J*(np.dot(theta[:][i],x)-x[i]*theta[i][i])
        #metropolis method
        r=np.exp(-valu)/(np.exp(-valu)+np.exp(valu))
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=1
        else:
            x[i]=-1

 
