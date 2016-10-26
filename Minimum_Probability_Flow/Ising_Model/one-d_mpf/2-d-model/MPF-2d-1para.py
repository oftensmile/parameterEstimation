#! /usr/bin/env python
#-*-coding:utf-8-*-
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
np.random.seed(1)
t_interval = 100
d, N_sample,N_model = 8,100,30 #124, 1000
N_remove=100
#parameter ( MPF+GD )
lr,eps =0.01, 1.0e-100
t_gd_max=300

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

if __name__ == '__main__':
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
    
    for t_gd in range(t_gd_max):
        gradK=0.0
        for M in data_matrix:
            x=np.copy(M.mat)
            gradK1=0.0
            for i1 in range(d):
                for i2 in range(d):
                    diff_E=-2*x[i1][i2]*(x[(i1+d-1)%d][i2]+x[(i1+1)%d][i2]+x[i1][(i2+d-1)%d]+x[i1][(i2+1)%d])
                    gradK1+=diff_E*np.exp(0.5*J_model*diff_E)/(d**2)
            gradK+=gradK1/N_sample
        J_model-=lr*gradK
        error=J_model-J_data
        print(abs(J_model-J_data))
