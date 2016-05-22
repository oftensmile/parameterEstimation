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
t_burn_emp, t_burn_model = 100, 10#10000, 100
t_interval = 10
#parameter ( System )
d, N_sample = 16,1000 #124, 1000
#parameter ( MPF+GD )
lr,eps =0.01, 1.0e-100
n_mfa = 100 #Number of the sample for Mean Field Aproximation.
t_gd_max=200 
def gen_mcmc(J,x=[] ):
    for i in range(d):
        #Heat Bath
        diff_E=2.0*J*x[i]*(x[(i+d-1)%d]+x[(i+1)%d])#E_new-E_old
        r=1.0/(1+np.exp(diff_E)) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x
"""
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
"""
"""
def calc_M(X=[[]]):
    n_bach=len(X)
    M=0.0
    for n in range(n_bach):
        xn=X[n]
        M+=np.sum(xn)/d
    M/=n_bach
    return M
"""
"""
def calc_C(X=[[]]):
    n_bach = len(X)
    corre_mean=0.0
    for n in range(n_bach):
        xn=X[n]
        corre=0.0
        for i in range(d):
            corre+=xn[i]*xn[(i+1)%d]/d
        corre_mean+=corre
    corre_mean/=n_bach
    return corre_mean
"""
########    MAIN    ########
#Generate sample-dist
J=1.1 # =theta_sample
x = np.random.uniform(-1,1,d)
x = np.array(np.sign(x))
for t_burn in range(t_burn_emp):
    x=np.copy(gen_mcmc(J,x))
#SAMPLING
for n in range(N_sample):
    for t in range(t_interval):
        x = np.copy(gen_mcmc(J,x))
    if(n==0):X_sample = np.copy(x)
    elif(n>0):X_sample=np.vstack((X_sample,np.copy(x)))

theta_model=1.5   #Initial Guess
print("#gd-step, abs-grad_likelihood, theta-error")
for t_gd in range(t_gd_max):
    #calc gradK of theta
    gradK=0.0
    n_bach=len(X_sample)
    for nin in range(n_bach):
        x_nin=X_sample[nin]
        gradK_nin=0.0
        #hamming distance = 1
        for hd in range(d):
            diff_delE_nin=-2.0*x_nin[hd]*(x_nin[(hd+d-1)%d]+x_nin[(hd+1)%d])
            diff_E_nin=diff_delE_nin*theta_model
            gradK_nin+=diff_delE_nin*np.exp(0.5*diff_E_nin)
        gradK+=gradK_nin
    gradK*=(1.0/n_bach)
    theta_model=np.copy(theta_model) - lr * gradK
    theta_diff=abs(theta_model-J)
    print(t_gd,np.abs(gradK),theta_diff)
print("#theta_true=",J,"theta_estimated=",theta_model)

