#! /usr/bin/env python
#-*-coding:utf-8-*-
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
import csv
from scipy.optimize import fsolve
from scipy.optimize import newton 
from scipy.optimize import minimize 
import math
n_estimation=3
np.random.seed(1)
t_interval = 10
#parameter ( System )
d, N_sample = 5,50 #124, 1000
N_remove=30
#parameter ( MPF+GD )
lr,eps =0.1, 1.0e-100
n_mfa = 50 #Number of the sample for Mean Field Aproximation.
t_gd_max=100 
def gen_mcmc(J,x=[] ):
    for i in range(d):
        #Heat Bath
        diff_E=2.0*J*x[i]*(x[(i+d-1)%d]+x[(i+1)%d])#E_new-E_old
        r=1.0/(1+np.exp(diff_E)) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x

def calc_C(X=[[]]):
    n_bach = len(X)
    corre_mean=0.0
    for n in range(n_bach):
        xn=X[n]
        corre=0.0
        for i in range(d):
            corre+=xn[i]*xn[(i+1)%d]
        corre_mean+=corre/n_bach
    return corre_mean

def Tk(J,k):
    l1=(2*np.cosh(J))**k
    l2=(2*np.sinh(J))**k
    return ( 0.5*(l1+l2) , 0.5*(l1-l2) )

#p(x_i=+1|x_1-i)
def gen_x_pofx(p_value):
    r=np.random.uniform()
    if(p_value>r):x_prop=1
    else:x_prop=-1
    return x_prop

def pofx_given_xprev(J,k,x_1,x_prev):
    ind_plus_prev=int(0.5*(1-x_prev)) #if same sign=>0
    ind_first_prev=int(0.5*(1-x_1*x_prev)) #if same sign=>0
    p=Tk(J,1)[ind_plus_prev] * Tk(J,d-k)[0] / Tk(J,d-k+1)[ind_first_prev]
    return p

def get_sample(j):
    X=np.zeros(d)
    #p(+)=p(-)=1/2
    X[0]=np.random.choice([-1,1])
    for k in range(1,d):
        p = pofx_given_xprev(j,k,X[0],X[k-1])
        X[k]=gen_x_pofx(p)
    return X

def Obfunc_1d_1para(J,g_data_sum):
    return -g_data_sum+(d*np.cosh(J)*np.sinh(J)*(np.cosh(J)**(-2+d)+np.sinh(J)**(-2+d)))/(np.cosh(J)**d+np.sinh(J)**d)

def myobj(J,i,X_sample):
    for n in range(N_sample)

if __name__ == '__main__':
    fname="sample"+str(N_sample)+"MCMC.dat"
    #f=open(fname,"w")
    #for nf in range(n_estimation):
    J_data=1.0 # =theta_sample
    #SAMPLING-Tmat
    c_mean_data=0.0
    for n in range(N_sample):
        x=get_sample(J_data)
        if(n==0):X_sample = np.copy(x)
        elif(n>0):X_sample=np.vstack((X_sample,np.copy(x)))
    corre_sample_mean=calc_C(X_sample) 
     
    for i in range(d):
        for n in range(N_sample):
            xn=np.copy(X_sample[n])

    J_newton=newton(Obfunc_1d_1para,0.5,args=(corre_sample_mean,))
   

    xi = np.array(np.sign(np.random.uniform(-1,1,d)))
    theta_model=2.0   #Initial Guess
    
    
    for t_gd in range(t_gd_max):
        for n_model in range(n_mfa+N_remove):
            for t in range(t_interval):
                xi = np.copy(gen_mcmc(theta_model,xi))
            if (n_model==N_remove):Xi_model = np.copy(xi)
            elif(n_model>N_remove):Xi_model = np.vstack((Xi_model,np.copy(xi)))
        corre_model_mean=calc_C(Xi_model)
        grad_likelihood=-corre_sample_mean+corre_model_mean
        theta_model=np.copy(theta_model)-lr*grad_likelihood
        theta_diff = theta_model-J_data
