#2016/05/19
##############
#   H = J*sum(xixj), J in R^1
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
d, N_sample = 16,1024 #124, 1000
#parameter ( MPF+GD )
#eps = 0.01
theta=[[1 if i==(j+1+d)%d or i==(j-1+d)%d else 0 for i in range(d)] for j in range(d)]
theta=np.array(theta)
#theta_model = np.arange(d*d)
#theta_model = np.reshape(theta_model,(d,d))#np.ones((d,d))
def gen_mcmc(J,x=[] ):
    for i in range(d):
        #Heat Bath
        diff_E=2.0*J*x[i]*(x[(i+d-1)%d]+x[(i+1)%d])#E_new-E_old
        r=1.0/(1+np.exp(diff_E)) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x
def calc_H(J,x=[]):
    e=0.0
    size=len(x)
    for i in range(size):
        e+=x[i]*x[(i+1)%size]
    e*=-J
    return e

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
def calc_heat_capacity(J,X=[[]]):
    n_bach=len(X)
    c1,c2=0.0,0.0
    for n in range(n_bach):
        xn=X[n]
        e_over_J=0.0
        for i in range(d):
            e_over_J+=xn[i]*xn[(i+1)%d]
        c2+=(e_over_J/d)**2
        c1+=e_over_J/d
    c=(1.0/n_bach)*(c2-c1**2)
    return c
def calc_twopint_corre(X=[[]]):
    #c1,c2=0.0,0.0
    c01,c0,c1=0.0,0.0,0.0
    n_bach=len(X)
    for n in range(n_bach):
        xn=X[n]
        c01+=(xn[0]*x[1])
        c0+=xn[0]
        c1+=xn[1]
    c=(1.0/n_bach)*(c01-c0*c1/n_bach)
    return c

########    MAIN    ########
#Generate sample
print("#bJ ,M_mean,E_mean/Jinv,d=",d,"N_sample=",N_sample)
"""
bJ_min, bJ_max = 0,4.0
d_bJ=0.01
n_bJ = int((bJ_max-bJ_min)/d_bJ)
for bJ in range(n_bJ):
    J = bJ_min+d_bJ*bJ
"""
#for nt in range(n_T):
    #Jinv=T_max -dT*nt
    #J=1.0/Jinv
dJ,nJ_max = 0.01,200
x = np.random.uniform(-1,1,d)
x = np.array(np.sign(x))
for t_burn in range(t_burn_emp):
    x = np.copy(gen_mcmc(0,x))
for nJ in range(nJ_max):
    J = dJ*nJ
    for t_burn in range(300):
        x = np.copy(gen_mcmc(J,x))
    #SAMPLING
    for n in range(N_sample):
        for t in range(t_interval):
            x = np.copy(gen_mcmc(J,x))
        if(n==0):X_sample = np.copy(gen_mcmc(J,x))
        elif(n>0):X_sample=np.vstack((X_sample,x))
    E_mean = calc_E(J,X_sample)
    M_mean = calc_M(X_sample)
    c2_mean=calc_twopint_corre(X_sample)
    print(J,"",np.abs(M_mean),"",E_mean,"",c2_mean)
