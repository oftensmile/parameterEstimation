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
d, N_sample = 4,640 #124, 1000
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
 
def gen_mcmc(J=[],x=[[]]):
    for i1 in range(d):
        for i2 in range(d):
            #Heat Bath
            diff_E=2.0*x[i1][i2]*(
                    x[(i1+d-1)%d][i2]*J[(i1+d-1)%d+i2*d]+x[(i1+1)%d][i2]*J[i1+i2*d]+x[i1][(i2+d-1)%d]*J[d*d+i1+(i2+d-1)%d*d]+x[i1][(i2+1)%d]*J[d*d+i1+i2*d])#E_new-E_old
            r=1.0/(1+np.exp(diff_E)) 
            R=np.random.uniform(0,1)
            if(R<=r):
                x[i1][i2]=x[i1][i2]*(-1)
    return x

def grad_obj(J,data_matrix):
    diff_expect=np.zeros(d*d*2)
    for M in data_matrix:
        x=np.copy(M.mat)
        for i1 in range(d):
            for i2 in range(d):
                dEi1i2=2*x[i1][i2]*(
                        x[(i1+d-1)%d][i2]*J[(i1+d-1)%d+i2*d]+x[(i1+1)%d][i2]*J[i1+i2*d]+
                        x[i1][(i2+d-1)%d]*J[d*d+i1+(i2+d-1)%d*d]+x[i1][(i2+1)%d]*J[d*d+i1+i2*d])
                
                dEi1pi2=2*x[(i1+1)%d][i2]*(
                        x[i1][i2]*J[i1+i2*d]+x[(i1+2)%d][i2]*J[(i1+1)%d+i2*d]+
                        x[(i1+1)%d][(i2+d-1)%d]*J[d*d+(i1+1)%d+(i2+d-1)%d*d]+x[(i1+1)%d][(i2+1)%d]*J[d*d+(i1+1)%d+i2*d])
                
                dEi1i2p=2*x[i1][(i2+1)%d]*(
                        x[(i1+d-1)%d][(i2+1)%d]*J[(i1+d-1)%d+(i2+1)%d*d]+x[(i1+1)%d][(i2+1)%d]*J[i1+(i2+1)%d*d]+
                        x[i1][i2]*J[d*d+i1+i2*d]+x[i1][(i2+2)%d]*J[d*d+i1+(i2+1)%d*d])
                
                diff_expect[i1+i2*d]+=-2*x[i1][i2]*x[(i1+1)%d][i2]*((1+np.exp(dEi1i2))**(-1)+(1+np.exp(dEi1pi2))**(-1) )/(N_sample*d**2)
                diff_expect[d*d+i1+i2*d]+=-2*x[i1][i2]*x[i1][(i2+1)%d]*((1+np.exp(dEi1i2))**(-1)+(1+np.exp(dEi1i2p))**(-1) )/(N_sample*d**2)
    return diff_expect 
if __name__ == '__main__':
    #fname="sample"+str(N_sample)+"MPF.dat"
    #f=open(fname,"w")
    #for nf in range(n_estimation):
    #Generate sample-dist
    x=np.random.choice([-1,1],(d,d))
    J_data=0.1*np.ones(d*d*2)
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
    J_model_root=root(grad_obj,0.01*np.ones(d*d*2),args=(data_matrix),method="hybr") 
    print("J_data=",J_data)
    print("J_model_root=",J_model_root.x)  
    print("diff=",np.sum(np.abs(J_model_root.x-J_data)) )  
        #f.write(str(J_diff)+"\n")
    #f.close()
