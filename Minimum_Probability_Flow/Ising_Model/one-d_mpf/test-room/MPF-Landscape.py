#! /usr/bin/env python
#-*-coding:utf-8-*-
#   E:= - \sum_{l}x_l x_{l+1} \theta_l
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
import csv 
np.random.seed(10)
J_true=0.5
t_interval = 50
d, N_sample =6,100 #124, 1000
N_remove = 100
lr,eps =0.1, 1.0e-100
t_gd_max=200 
bins=0.025

class State:
    def __init__(self,index, degree,state=[]):
        self.index=index    #This is not sorted order, but the decimal value of it.
        self.degree=degree
        self.state=np.copy(state)

def convert_binary_to_decimal(x=[]):
    #supportce that entry of the x is -1 or 1.
    size_x=len(x)
    decimal=0
    for i in range(size_x):
        decimal+=int(0.5*(x[i]+1) * 2**i )
    return decimal

def convert_decimal_to_binary(x):
    y=np.zeros(d)
    b=x
    y[0]=int(2*(x%2-0.5))
    a=x%2
    for l in range(1,d):
        b=b-a*2**(l-1)
        a=np.sign(b%(2**(l+1)))
        y[l]=int(2*(a-0.5))
    return y

def build_state_dist(X_sample=[[]]):
    global state_dist
    i=0
    address=[]
    for xn in X_sample:
        index=convert_binary_to_decimal(xn)
        if(len(address)==0):
            state_dist=[State(index,1,np.copy(xn))]
            address=np.append(address,index)
        elif(len(address)>0 and len(np.where(address==index)[0])==0):
            state_dist=np.append(state_dist, State(index,1,np.copy(xn)) )
            address=np.append(address,index)
        elif(len(address)>0 and len(np.where(address!=index)[0])>0):
            index2=np.where(address==index)[0][0]
            state_dist[index2].degree+=1
        i+=1

def gen_mcmc(J,x=[]):
    for i in range(d):
        diff_E=2.0*x[i]*J*(x[(i+1)%d]+x[(i+d-1)%d])
        r=1.0/(1+np.exp(diff_E)) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x

def prob_trans(th,j,x=[]):
    pout=1.0/(1.0 + np.exp( 2.0*th*x[j]*(x[(j+1)%d]+x[(j+d-1)%d]))) / d
    #   This shape must be propotional to r in gen_mcmc
    return  pout

def get_state_binary_represent(size):
    list_decimal=np.ones(size)*(-1)
    for q in range(size):
        list_decimal[q]=convert_binary_to_decimal(np.copy(state_dist[q].state))
    return list_decimal

def get_samples():
    x = np.random.choice([-1,1],d)
    for n in range(N_sample+N_remove):
        for t in range(t_interval):
            x = np.copy(gen_mcmc(J_true,x))
        if(n==N_remove):
            x_new=np.copy(x)
            X_sample = x_new
        elif(n>N_remove):
            x_new=np.copy(x)
            X_sample=np.vstack((X_sample,x_new))
    return X_sample

if __name__ =='__main__':
    #   get data
    X_sample=get_samples()
    #  create list object,state_dist.
    build_state_dist(X_sample)
    size_of_sample=len(state_dist)
    list_decimal=get_state_binary_represent(size_of_sample)
    #   calculation of ofjective function
    theta_slice=np.arange(-1.0,2.0,bins)
    MPF_of_th=0.0
    for th in theta_slice:
        MPF_of_th_old=MPF_of_th
        prob1=np.zeros(size_of_sample)
        prob0=np.zeros(size_of_sample)
        for l in range(size_of_sample):
            xl=np.copy(state_dist[l].state)
            index_xl=convert_binary_to_decimal(xl)
            prob0[l]=state_dist[l].degree/N_sample
            prob1[l]=state_dist[l].degree/N_sample
            for j in range(d):
                prob1[l]+= -state_dist[l].degree/N_sample * prob_trans(th,j,xl)
                xl[j]*=-1
                index_neighbor=convert_binary_to_decimal(xl)
                if( len(np.where(list_decimal==index_neighbor)[0]) > 0 ):
                    id2=np.where(list_decimal==index_neighbor)[0][0] 
                    prob1[l]+= state_dist[id2].degree/N_sample * prob_trans(th,j,xl)
                xl[j]*=-1
        MPF_of_th=np.dot(np.log(prob1),prob0)
        del_MPF = (MPF_of_th-MPF_of_th_old)/bins
        if(th==theta_slice[0]):del_MPF=0
        print(th,MPF_of_th,del_MPF,np.sum(prob1))
