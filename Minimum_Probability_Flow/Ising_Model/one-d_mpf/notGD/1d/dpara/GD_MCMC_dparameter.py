#! /usr/bin/env python
#-*-coding:utf-8-*-
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
import csv
from scipy.optimize import fsolve
from scipy.optimize import minimize 
from scipy.optimize import root 
import math
n_estimation=3
np.random.seed(1)
t_interval = 30
#parameter ( System )
d, N_sample = 5,100 #124, 1000
N_remove=40
#parameter ( MPF+GD )
lr,eps =0.1, 1.0e-100
n_mfa = 50 #Number of the sample for Mean Field Aproximation.
t_gd_max=80 
def gen_mcmc(J=[],x=[]):
    for i in range(d):
        #Heat Bath
        diff_E=2.0*x[i]*(J[(i+d-1)%d]*x[(i+d-1)%d]+J[i]*x[(i+1)%d])#E_new-E_old
        r=1.0/(1+np.exp(diff_E)) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x

def calc_C(X=[[]]):
    n_bach = len(X)
    corre_mean=np.zeros(d)
    for n in range(n_bach):
        xn=X[n]
        for i in range(d):
            #corre+=xn[i]*xn[(i+1)%d]/d
            corre_mean[i]+=xn[i]*xn[(i+1)%d]/n_bach
    return corre_mean

def mcmc_object(J=[],corre_data=[]):
    x=np.random.choice([-1,1],d)
    for n in range(N_sample+N_remove):
        for t in range(t_interval):
            x = np.copy(gen_mcmc(J,x))
        if(n==N_remove):X_sample = np.copy(x)
        elif(n>N_remove):X_sample=np.vstack((X_sample,np.copy(x)))
    corre_sample_mean=calc_C(X_sample)
    return corre_sample_mean-corre_data


if __name__ == '__main__':
    #fname="sample"+str(N_sample)+"MCMC.dat"
    #f=open(fname,"w")
    #for nf in range(n_estimation):
    J_data=np.random.rand(d) # =theta_sample
    #SAMPLING
    x=np.random.choice([-1,1],d)
    for n in range(N_sample+N_remove):
        for t in range(t_interval):
            x = np.copy(gen_mcmc(J_data,x))
        if(n==N_remove):X_sample = np.copy(x)
        elif(n>N_remove):X_sample=np.vstack((X_sample,np.copy(x)))
    corre_sample_mean=calc_C(X_sample)
    print("corre_sample_mean=\n",corre_sample_mean)
    J_model=root(mcmc_object,0.1*np.ones(d),args=(corre_sample_mean),method="hybr")
    J_model_list=J_model.x
    
    print("#J_model= \n",J_model_list)
    print("#J_data= \n",J_data)
    diff=(J_data-J_model_list)
    print("#J_model-Jdata= \n",diff)
    #f.close()
