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
d, N_sample = 32,300 #124, 1000
#parameter ( MPF+GD )
lr,eps =0.1, 1.0e-100
n_mfa = 100 #Number of the sample for Mean Field Aproximation.
t_gd_max=300
#theta=[[1 if i==(j+1+d)%d or i==(j-1+d)%d else 0 for i in range(d)] for j in range(d)]
#theta=np.array(theta)
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
#GD+MFA
corre_sample_mean=calc_C(X_sample) 
#Generate model-dist
xi = np.array(np.sign(np.random.uniform(-1,1,d)))
theta_model=2.0   #Initial Guess
#Burn-in
#for t_burn in range(t_burn_model*10):
#    xi=np.copy(gen_mcmc(theta_model,xi))
print("#gd-step, abs-grad_likelihood, theta-error")
for t_gd in range(t_gd_max):
    for t_burn in range(t_burn_model*10):
        xi = np.copy(gen_mcmc(theta_model,xi))
    for n_model in range(n_mfa):
        for t in range(t_interval):
            xi = np.copy(gen_mcmc(theta_model,xi))
        if (n_model==0):Xi_model = np.copy(xi)
        elif(n_model>0):Xi_model = np.vstack((Xi_model,np.copy(xi)))
    corre_model_mean=calc_C(Xi_model)
    grad_likelihood=-corre_sample_mean+corre_model_mean
    theta_model=np.copy(theta_model)-lr*grad_likelihood
    #theta_model=np.copy(theta_model)-lr*(1.0/np.log(t_gd+1.7))*grad_likelihood
    theta_diff = np.abs(theta_model-J)
    print(t_gd,np.abs(grad_likelihood),theta_diff)
    #This stopping condition doesn't work.
    #if(grad_likelihood<eps):
    #    break
print("#theta_true=",J,"theta_estimated=",theta_model)
