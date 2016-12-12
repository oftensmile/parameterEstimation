#! /usr/bin/env python
#-*-coding:utf-8-*-
import numpy as np
import time 
from scipy import linalg
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

def mcmc_object(J=[],corre_data=[],X_sample=[[]]):
    for m in range(N_sample):
        x_init=np.copy(X_sample[m])
        x_new_for_mcmc=np.copy(gen_mcmc(J,x_init))#This update is possible to generate any state.
        if(m==0):X_cd=np.copy(x_new_for_mcmc)  
        else:
            X_cd=np.vstack((X_cd,np.copy(x)))
    corre_sample_mean=calc_C(X_cd)
    return corre_sample_mean-corre_data


if __name__ == '__main__':
    estimation_list=[5,10,20]
    N_sample=100
    fname="sample"+str(N_sample)+"-estimation-n-subset-naiveCD.dat"
    f=open(fname,"w")
    for n_estimation in estimation_list:
        list_J_model=np.zeros(n_estimation)
        J_data=np.random.rand(d) # =theta_sample
        for nf in range(n_estimation):
            #SAMPLING
            x=np.random.choice([-1,1],d)
            for n in range(N_sample*2+N_remove):
                for t in range(t_interval):
                     x = np.copy(gen_mcmc(J_data,x))
                if(n==N_remove):X_set1 = np.copy(x)
                elif(n>N_remove and N_remove+N_sample>n):
                    X_set1=np.vstack((X_set1,np.copy(x)))
                elif(n==N_remove+N_sample):X_set2 = np.copy(x)
                elif(n>N_remove+N_sample):
                    X_set2=np.vstack((X_set2,np.copy(x)))
            correlation_data=calc_C(X_set1)
            J_model=1.5*np.random.rand(d)
            for t_gd in range(t_gd_max):
                gradl=np.zeros(d)
                for m in range(N_sample):
                    x_init=np.copy(X_set2[m])
                    x_new_for_mcmc=gen_mcmc(J_model,x_init)
                    if(m==0):X_set_model=np.copy(x_new_for_mcmc)
                    else:X_set_model=np.vstack((X_set_model,x_new_for_mcmc))
                correlation_model=calc_C(X_set_model)
                J_model=J_model-lr*(correlation_model-correlation_data) 
            error=sum(J_model-J_data)
            list_J_model[nf]=error
        mean=np.mean(list_J_model)
        std=np.std(list_J_model)/np.sqrt(n_estimation)
        f.write(str(n_estimation)+"  "+str(mean)+"  "+str(std)+"\n")
    f.close()
