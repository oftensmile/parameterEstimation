#! /usr/bin/env python
#-*-coding:utf-8-*-
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
import csv
np.random.seed(1)
n_estimation=300
d, N_sample = 16,2 #124, 1000
N_remove = 100
lr=0.05
t_gd_max=500 
def gen_mcmc(J,x=[] ):
    for i in range(d):
        #Heat Bath
        diff_E=2.0*J*x[i]*(x[(i+d-1)%d]+x[(i+1)%d])#E_new-E_old
        r=1.0/(1+np.exp(diff_E)) 
        #r=np.exp(-diff_E) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x

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


if __name__ == '__main__':
    #fname="sample"+str(N_sample)+"MPF.dat"
    #f=open(fname,"w")
    N_sample=30
    n_estimation=1
    for nf in range(n_estimation):
        #Generate sample-dist
        #J_data=2.0001 # =theta_sample
        J_data=1.01 # =theta_sample
        x = np.random.choice([-1,1],d)
        #SAMPLING-Tmat
        for n in range(N_sample):
            x=get_sample(J_data)
            if(n==0):X_sample = np.copy(x)
            elif(n>0):X_sample=np.vstack((X_sample,np.copy(x)))

        n_bach=len(X_sample)
        J_model=2.0   #Initial Guess
        error_vec=np.zeros(t_gd_max)
        for t_gd in range(t_gd_max):
            #calc gradK of theta
            gradK=0.0
            for sample in X_sample:
                #x_nin=np.reshep(np.copy(sample),(d,d)
                x_nin=np.copy(sample)
                gradK_nin=0.0
                #hamming distance = 1
                for hd in range(d):
                    diff_delE_nin=-2.0*x_nin[hd]*(x_nin[(hd+d-1)%d]+x_nin[(hd+1)%d])
                    #diff_E_nin=diff_delE_nin*J_model
                    #gradK_nin+=diff_delE_nin*np.exp(0.5*diff_E_nin)/d
                    gradK_nin+=diff_delE_nin*np.exp(0.5*J_model*diff_delE_nin)/d


                gradK+=gradK_nin/n_bach
            J_model-= lr * gradK
            J_diff=J_model-J_data
            error_vec[t_gd]=J_diff
        
        error_vec_cd=np.zeros(t_gd_max)
        J_model_cd=2.0
        for t_gd in range(t_gd_max):
            diff_expect=0.0#np.zeros(d)
            for m in range(N_sample):
                x_m=np.copy(X_sample[m])
                for l in range(d):
                    diff_E=2*x_m[l]*(x_m[(l+1)%d]+x_m[(l-1+d)%d])
                    diff_expect+=( - diff_E * (d*(1+np.exp(J_model_cd*diff_E)))**(-1))/N_sample
            J_model_cd-=d*lr*diff_expect
            error_cd=J_model_cd - J_data
            error_vec_cd[t_gd]=error_cd
    ptitle="Learning Curve, d="+str(d)+", N="+str(N_sample)
    plt.plot(error_vec,label="MPF")
    plt.plot(error_vec_cd,label="CD(without MCMC)")
    plt.xlabel("epoch",fontsize=18)
    plt.ylabel("error",fontsize=18)
    plt.title(ptitle,fontsize=18)
    plt.legend(fontsize=18)
    plt.show()    
