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
import matplotlib.pyplot as plt
import math
n_estimation=3
np.random.seed(1)
t_interval = 30
#parameter ( System )
d, N_sample = 16,100 #124, 1000
N_remove=100
N_mcmc=50
#parameter ( MPF+GD )
lr,eps =0.03, 1.0e-100
n_mfa = 50 #Number of the sample for Mean Field Aproximation.
t_gd_max=400 
def gen_mcmc(J=[],x=[]):
    for i in range(d):
        #Heat Bath
        diff_E=2.0*x[i]*(J[(i+d-1)%d]*x[(i+d-1)%d]+J[i]*x[(i+1)%d])#E_new-E_old
        r=1.0/(1+np.exp(diff_E)) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x

def gen_mcmc_single(J,x=[]):
    i=np.random.randint(d)
    #Heat Bath
    diff_E=2.0*x[i]*(J[(i+d-1)%d]*x[(i+d-1)%d]+J[i]*x[(i+1)%d])#E_new-E_old
    #diff_E=2.0*x[index]*J*(x[(d+1+index)%d]+x[(index+d-1)%d])
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
    #print("corre_sample_mean=\n",corre_sample_mean)
    J_model_init=np.random.uniform(0,2,d)
    J_model1=np.copy(J_model_init)
    #J_model=root(mcmc_object,np.copy(J_model_init),args=(corre_sample_mean),method="hybr")
    error_list1=np.zeros(t_gd_max) 
    for t_gd in range(t_gd_max):
        gradl=np.zeros(d)
        for m in range(N_mcmc+N_remove):
            x_new_for_mcmc=np.copy(np.random.choice([-1,1],d) )
            for t in range(t_interval):
                x_new_for_mcmc=gen_mcmc(J_model1,x_init)
            if(m==N_remove):X_set_model=np.copy(x_new_for_mcmc)
            elif(m>N_remove):X_set_model=np.vstack((X_set_model,x_new_for_mcmc))
        correlation_model=calc_C(X_set_model)
        J_model1=J_model1-lr*(correlation_model-corre_sample_mean) 
        error=sum(J_model1-J_data)
        print(error)
        error_list1[t_gd]=error
    L=np.int(np.sqrt(d))
    J_model_mat=np.reshape(J_model1,(L,L)) 
    J_data_mat=np.reshape(J_data,(L,L)) 
    plt.figure()
    plt.subplot(131)
    plt.imshow(J_model_mat, interpolation='nearest')
    plt.colorbar()
    plt.title("J_model")
    plt.subplot(132)
    plt.imshow(J_data_mat, interpolation='nearest')
    plt.colorbar()
    plt.title("J_data")
    plt.subplot(133)
    plt.imshow(J_model_mat-J_data_mat, interpolation='nearest')
    plt.colorbar()
    plt.title("J_model_mat-J_data_mat")
    plt.clim(-0.01,1)
    plt.show()
     #bins=np.arange(1,d+1)
    #bar_width=0.2
    #plt.bar(bins,J_data,color="blue",width=bar_width,label="$\it{J_{data}}$",align="center")
    #plt.bar(bins+bar_width,J_model,color="red",width=bar_width,label="$\it{J_{moel}}$",align="center")
    #plt.bar(bins+2*bar_width,init_theta,color="green",width=bar_width,label="initial",align="center")
    #plt.bar(bins+2*bar_width,J_model_init,color="gray",width=bar_width,label="$\it{J_{moel}};initial$",align="center")
    #plt.title("Maximum Likelihood Estimation",fontsize=22)
    #plt.title("Contrastive Divergence(k=1)",fontsize=22)
    #plt.xlabel("i=1,2,...,16",fontsize=18)
    #plt.ylabel("J",fontsize=18)

