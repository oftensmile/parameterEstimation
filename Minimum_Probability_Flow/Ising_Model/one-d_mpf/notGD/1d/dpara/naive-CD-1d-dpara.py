#! /usr/bin/env python
#-*-coding:utf-8-*-
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.optimize import root 
import math
n_estimation=3
np.random.seed(1)
t_interval = 30
#parameter ( System )
d, N_sample = 16,100 #124, 1000
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
    #fname="sample"+str(N_sample)+"MCMC.dat"
    #f=open(fname,"w")
    #for nf in range(n_estimation):
    J_data=np.random.uniform(0,1,d) # =theta_sample
    #SAMPLING
    x=np.random.choice([-1,1],d)
    for n in range(N_sample+N_remove):
        for t in range(t_interval):
            x = np.copy(gen_mcmc(J_data,x))
        if(n==N_remove):X_sample = np.copy(x)
        elif(n>N_remove):X_sample=np.vstack((X_sample,np.copy(x)))
    corre_sample_mean=calc_C(X_sample)
    J_model_init=np.random.uniform(0,2,d)
    J_lm=root(mcmc_object,J_model_init,args=(corre_sample_mean,X_sample),method="lm")
    J_groyden1=root(mcmc_object,0.1*np.ones(d),args=(corre_sample_mean,X_sample),method="broyden1")
    """
    J_groyden2=root(mcmc_object,0.1*np.ones(d),args=(corre_sample_mean,X_sample),method="broyden2")
    J_anderson=root(mcmc_object,0.1*np.ones(d),args=(corre_sample_mean,X_sample),method="anderson")
    J_limix=root(mcmc_object,0.1*np.ones(d),args=(corre_sample_mean,X_sample),method="linearmixing")
    J_diag=root(mcmc_object,0.1*np.ones(d),args=(corre_sample_mean,X_sample),method="diagbroyden")
    J_krylov=root(mcmc_object,0.1*np.ones(d),args=(corre_sample_mean,X_sample),method="krylov")
    """
    diff=(J_lm.x-J_data)
    print("diff_J_lm=\n",diff)
    bins=np.arange(1,d+1)
    bar_width=0.2
    plt.bar(bins,J_data,color="blue",width=bar_width,label="$\it{J_{data}}$",align="center")
    plt.bar(bins+bar_width,J_lm.x,color="red",width=bar_width,label="$\it{J_{moel}}$",align="center")
    #plt.bar(bins+2*bar_width,init_theta,color="green",width=bar_width,label="initial",align="center")
    plt.bar(bins+2*bar_width,J_model_init,color="gray",width=bar_width,label="$\it{J_{moel}};initial$",align="center")
    plt.legend(fontsize=18)
    plt.title("Maximum Likelihood Estimation",fontsize=22)
    plt.xlabel("i=1,2,...,16",fontsize=18)
    plt.ylabel("J",fontsize=18)
    plt.show()    
    """
    diff=(J_groyden1.x-J_data)
    print("diff J_groyden1=\n",diff)
    diff=(J_groyden2.x-J_data)
    print("diff J_groyden2=\n",diff)
    diff=(J_anderson.x-J_data)
    print("#diff J_anderson= \n",diff)
    diff=(J_limix-J_data)
    print("#diff .J_limix= \n",diff)
    diff=(J_diag-J_data)
    print("#diff J_di= \n",diff)
    diff=(J_krylov-J_data)
    print("#diff J_krylov= \n",diff)
    """
    #f.close()
