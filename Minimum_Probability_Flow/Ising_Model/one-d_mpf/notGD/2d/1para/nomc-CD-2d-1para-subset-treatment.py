#! /usr/bin/env python
#-*-coding:utf-8-*-
import numpy as np
import time 
from scipy import linalg
from scipy.optimize import root
import matplotlib.pyplot as plt
import csv
np.random.seed(0)
n_estimation=300
#dT=T_max/n_T 
t_interval = 100
d, N_sample = 8,100 #124, 1000
N_remove = 100
lr = 0.1
#n_mfa = 100 #Number of the sample for Mean Field Aproximation.
t_gd_max=100 
class Matrices():
    def __init__(self,mat):
        self.mat=mat

def build_matrix(x=[[]]):
    global matrices
    matrices=tuple(Matrices(x) for i in range(1))
 
def gen_mcmc(J,x=[[]]):
    for i1 in range(d):
        for i2 in range(d):
            #Heat Bath
            diff_E=2.0*J*x[i1][i2]*(x[(i1+d-1)%d][i2]+x[(i1+1)%d][i2]+x[i1][(i2+d-1)%d]+x[i1][(i2+1)%d])#E_new-E_old
            r=1.0/(1+np.exp(diff_E)) 
            R=np.random.uniform(0,1)
            if(R<=r):
                x[i1][i2]=x[i1][i2]*(-1)
    return x

def grad_obj(J,data_matrix):
    diff_expect=0.0
    for M in data_matrix:
        x=np.copy(M.mat)
        for i1 in range(d):
            for i2 in range(d):
                diff_E=2*x[i1][i2]*(x[(i1+d-1)%d][i2]+x[(i1+1)%d][i2]+x[i1][(i2+d-1)%d]+x[i1][(i2+1)%d])
                diff_expect+=( -diff_E/(1+np.exp(J*diff_E)) )/(N_sample*d**2)
    return diff_expect 
if __name__ == '__main__':
    #fname="sample"+str(N_sample)+"MPF.dat"
    #f=open(fname,"w")
    #for nf in range(n_estimation):
    #Generate sample-dist
    x=np.random.choice([-1,1],(d,d))
    J_data,J_model=0.5,0.1
    for n in range(N_sample+N_remove):
        if(n<N_remove):
            x=np.copy(gen_mcmc(J_data,x))
        elif(n==N_remove):
            x=np.copy(gen_mcmc(J_data,x))
            build_matrix(x)
            data_matrix=matrices
        else:
            for t in range(t_interval):
                x=np.copy(gen_mcmc(J_data,x))
            data_matrix=np.append(data_matrix,Matrices(x))
    J_model_root=root(grad_obj,0.01,args=(data_matrix),method="hybr") 
    print("J_data=",J_data)
    print("J_model_root=",J_model_root.x)  
        #f.write(str(J_diff)+"\n")
    #f.close()
