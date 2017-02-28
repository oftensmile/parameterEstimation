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
d,dh, N_sample = 5,5,50 #124, 1000
N_remove=50
#parameter ( MPF+GD )
lr,eps =0.1,0.1 
k_max=10
t_gd_max=100 
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
#it seems better to update the parameters within this function
def sampling_phase(eps,k_max,N_sample,a,b,W,X_sample):
    grad_a=np.zeros(d)
    grad_b=np.zeros(dh)
    grad_W=np.ones((d,dh))
    for n in range(N_sample):
        v=np.copy(X_sample[n])
        v0=np.copy(X_sample[n])
        h=np.zeros(dh)
        #   sampling h, initial
        h_e=b+np.dot(v,W)
        for j in range(dh):
            r=np.random.uniform(0,1)
            if( r< 1/(1+np.exp(2*(-h_e[j]))) ):
                h[j]=1
            else:
                h[j]=-1
        h1=np.copy(h)
        #   sampling k steps
        for k in range(k_max):
            #   sampling v
            v_e=a+np.dot(W,h)
            for i in range(d):
                r=np.random.uniform(0,1)
                if(r<(1+np.exp(2*(-v_e[i])))**(-1)):
                    v[i]=1
                else:
                    v[i]=-1
            #   sampling h
            h_e=b+np.dot(v,W)
            for j in range(dh):
                r=np.random.uniform(0,1)
                if(r<(1+np.exp(2*(-h_e[j])))**(-1)):
                    h[j]=1
                else:
                    h[j]=-1
        grad_a=grad_a+(v0-v)/N_sample
        grad_b=grad_b+(h1-h)/N_sample
        grad_W=grad_W+( np.dot(np.matrix(v0).T,np.matrix(h1)) - np.dot(np.matrix(v).T,np.matrix(h)) )/N_sample
        #print(np.dot(v,v0)/d)
    a=np.copy(a)+eps*grad_a
    b=np.copy(b)+eps*grad_b
    W=np.copy(W)+eps*grad_W
    return (np.copy(a),np.copy(b),np.copy(v),np.copy(W) )

if __name__ == '__main__':
    J_data=np.random.rand(d) # =theta_sample
    x=np.random.choice([-1,1],d)
    for n in range(N_sample+N_remove):
        for t in range(t_interval):
            x = np.copy(gen_mcmc(J_data,x))
        if(n==N_remove):X_sample = np.copy(x)
        elif(n>N_remove):X_sample=np.vstack((X_sample,np.copy(x)))
    a=np.ones(d)*0.1#/d
    b=np.ones(dh)*0.1#/d
    W=np.zeros((d,dh))#*0.5
    for i in range(d):
        W[i][i]=1.0
        W[i][(i-1+dh)%dh]=1.0
    #W=W+W.T
    for t in range(t_gd_max):
        a,b,v,W = sampling_phase(eps,k_max,N_sample,a,b,W,X_sample)
        # only W[i][i] and W[i][i-1] are notribial , lest of the elements are zeros.
        tempW1=np.zeros(d)
        tempW2=np.zeros(d)
        for i in range(d):
            tempW1[i]=W[i][i]
            tempW2[i]=W[i][(i-1+dh)%dh]
        W=np.zeros((d,dh))
        for i in range(d):
            W[i][i]=tempW1[i]
            W[i][(i-1+dh)%dh]=tempW2[i]
    
    tanh=np.zeros(dh)
    for n in range(N_sample):
        xn=np.copy(X_sample[n])
        tanh=tanh+np.tanh(b+np.dot(xn,W))/N_sample
    one_tanh2=np.zeros(dh)-tanh*tanh
    diag_one_tanh2=np.diag(one_tanh2)
    J_model=-np.dot(np.matrix(W),np.dot(diag_one_tanh2,np.matrix(W).T) )
    print("J_data=\n",J_data)
    print("a=\n",a)
    print("b=\n",b)
    print("w=\n",W)
    print("J=\n",J_model)
    print("X_sample[N_sample-1]=\n",X_sample[N_sample-1])
    print("v=\n",v)

