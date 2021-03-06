#2016/05/19
##############
#   H = -J*sum(xixj), J in R^1
##############
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
import csv
np.random.seed(0)
#parameter ( Model )
T_max=1.2
#Temperature Dependance
#= J^-1=kT/J=T/Tc, Tc=J/k=1
n_T=100
dT=T_max/n_T 

#parameter ( MCMC )
t_burn_emp, t_burn_model = 1000, 10#10000, 100
t_interval = 10
#parameter ( System )
d, N_sample = 16,8192 #124, 1000
#parameter ( MPF+GD )
#eps = 0.01
#theta=[[1 if i==(j+1+d)%d or i==(j-1+d)%d else 0 for i in range(d)] for j in range(d)]
#theta=np.array(theta)
#theta_model = np.arange(d*d)
#theta_model = np.reshape(theta_model,(d,d))#np.ones((d,d))
def gen_mcmc(J,x=[] ):
    for i in range(d):
        #Metropolice
        E_diff=-(-1)*2.0*J*x[i]*( x[(i+d-1)%d]+x[(i+1)%d])#positive sign
        r=np.exp(-E_diff)
        R=np.random.uniform(0,1)
        #r=np.exp(-valu)/(np.exp(-valu)+np.exp(valu))
        if(R<=r):
            x[i]=x[i]*(-1)
    return x

def calc_E(J,X=[[]]):
    n_bach=len(X)
    E=0.0
    for n in range(n_bach):
        xn=X[n]
        e=0.0
        for i in range(d):
            e+=xn[i]*xn[(i+1)%d]
        e*=J*(1.0/d)
        E+=e
    E/=n_bach
    return E

def calc_M(X=[[]]):
    n_bach=len(X)
    M=0.0
    for n in range(n_bach):
        xn=X[n]
        M+=np.sum(xn)/d
    M/=n_bach
    return M

########    MAIN    ########
#Generate sample
print("#Jinv ,M_mean,E_mean/Jinv,d=",d,"N_sample=",N_sample)
for nt in range(n_T):
    Jinv=T_max -dT*nt
    J=1.0/Jinv
    #x = np.ones(d)
    x = np.random.uniform(-1,1,d)
    x = np.array(np.sign(x))
    #BURN-IN 
    for t_burn in range(t_burn_emp):
        x = np.copy(gen_mcmc(J,x))
    #SAMPLING
    for n in range(N_sample):
        for t in range(t_interval):
            x = np.copy(gen_mcmc(J,x))
        if(n==0):X_sample = np.copy(gen_mcmc(J,x))
        elif(n>0):X_sample=np.vstack((X_sample,x))
    E_mean = calc_E(J,X_sample)
    M_mean = calc_M(X_sample)
    print(Jinv,"",np.abs(M_mean),"",np.abs(E_mean*Jinv))
