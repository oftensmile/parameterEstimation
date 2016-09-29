#! /usr/bin/env python
#-*-coding:utf-8-*-
#   E:= - \sum_{l}x_l x_{l+1} \theta_l
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
import csv 
np.random.seed(10)
#parameter ( MCMC )
t_interval = 40
d, N_sample =6,50 #124, 1000
num_mcmc_sample=100
J_true=0.1
N_remove = 100
lr,eps =0.1, 1.0e-100
t_gd_max=200 
bins=0.025

class State:
    def __init__(self,index, degree,state=[]):
        self.index=index    #This is not sorted order, but the decimal value of it.
        self.degree=degree
        self.state=np.copy(state)

class P_in_to_out:
    def __init__(self,index,prob,state=[]):
        self.index=index    #This is not sorted order, but the decimal value of it.
        self.prob=prob
        self.state=np.copy(state)

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

def calc_E(th,x=[]):
    E=0.0
    for i in range(d):
        E+=-th*x[i]*x[(i+1)%d]
    return E

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
    #CD0_of_th_m=th*sum_correlation_data_vec[m]-np.log( (2*np.cosh(th))**d + (2*np.sinh(th))**d)
    for th in theta_slice:
        prob0=np.zeros(size_of_sample)
        prob1=np.zeros(size_of_sample)
        prob1_outdata_record=[]
        kl0=np.zeros(size_of_sample)
        KL0,KL1=0.0,0.0
        Z_th=(2*np.cosh(th))**d + (2*np.sinh(th))**d
        for l in range(size_of_sample):
            xl=np.copy(state_dist[l].state)
            E_xl=calc_E(th,xl)
            index_xl=convert_binary_to_decimal(xl)
            prob0[l]=state_dist[l].degree/N_sample
            prob1[l]=state_dist[l].degree/N_sample
            prob_infty=np.exp(-E_xl)/Z_th
            kl0[l]=prob0[l]*np.log(prob0[l] /prob_infty)
            for j in range(d):
                prob1[l]+= -state_dist[l].degree/N_sample * prob_trans(th,j,xl)
                xl[j]*=-1
                index_neighbor = convert_binary_to_decimal(xl)
                size_neighbor = len(np.where(list_decimal==index_neighbor)[0])
                if( size_neighbor > 0 ):#this state is included in the data
                    id2=np.where(list_decimal==index_neighbor)[0][0] 
                    prob1[l]+= state_dist[id2].degree/N_sample * prob_trans(th,j,xl)
                elif(size_neighbor== 0):#this state is not included in the data
                    #This state have note appear in the 
                    #Record in the 
                    prob1_into_temp=(1/d-prob_trans(th,j,xl))*prob0[l]# data to out of data
                    if(len(prob1_outdata_record)==0):
                        prob1_outdata_record=[index_neighbor]
                        prob1_outofdata=[P_in_to_out(index_neighbor,prob1_into_temp,xl)]
                    elif(len(prob1_outdata_record)>0):
                        size_init=len(np.where(prob1_outdata_record==index_neighbor)[0])
                        if(size_init==0):
                            prob1_outdata_record=np.append(prob1_outdata_record,index_neighbor)
                            prob1_outofdata=np.append(prob1_outofdata,P_in_to_out(index_neighbor,prob1_into_temp,xl))
                        elif(size_init>0):
                            id3=np.where(prob1_outdata_record==index_neighbor)[0][0]
 
                            prob1_outofdata[id3].prob=prob1_outofdata[id3].prob+prob1_into_temp
                xl[j]*=-1
                
            KL0+=kl0[l]
        len_prob1_outdata_reco=len(prob1_outdata_record)
        for h in range(len_prob1_outdata_reco):
            x_h=np.copy(prob1_outofdata[h].state)
            E_h=calc_E(th,x_h)
            prob_infty=np.exp(-E_h)/Z_th
            KL1+=prob1_outofdata[h].prob * np.log( prob1_outofdata[h].prob / prob_infty)
        CD=KL0-KL1
        print(th,CD,KL0,KL1)

