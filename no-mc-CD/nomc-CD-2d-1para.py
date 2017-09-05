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

def calc_E(J,x=[[]]):
    erg=0
    for i1 in range(d):
        for i2 in range(d):
            erg+=x[i1][i2]*(x[(i1+d-1)%d][i2]+x[(i1+1)%d][i2]+x[i1][(i2+d-1)%d]+x[i1][(i2+1)%d])
    return 0.5*erg

def get_E_C(J):
    #x=np.ones((d,d))
    x=np.random.choice([-1,1],(d,d))
    E,E2=0.0,0.0
    for n in range(N_sample+N_remove):
        if(n<N_remove):
            x=np.copy(gen_mcmc(J,x))
        else:
            for t in range(t_interval):
                x=np.copy(gen_mcmc(J,x))
            E_temp=calc_E(J,x)
            E+=E_temp/N_sample
            E2+=E_temp**2 / N_sample
    return (E,-E2)

def calc_C(x=[[]]):
    c=0.0
    for i1 in range(d):
        for i2 in range(d):
            c+=x[i1][i2]*(x[(i1+d-1)%d][i2]+x[(i1+1)%d][i2]+x[i1][(i2+d-1)%d]+x[i1][(i2+1)%d])
    return 0.5*c

if __name__ == '__main__':
    x=np.random.choice([-1,1],(d,d))
    J_data,J_model=0.5,0.1
    C_data,C_model=0.0,0.0
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
            C_data+=calc_C(x)/N_sample
    
    for t_gd in range(t_gd_max):
        diff_expect=0.0
        for M in data_matrix:
            x=np.copy(M.mat)
            for i1 in range(d):
                for i2 in range(d):
                    diff_E=2*x[i1][i2]*(x[(i1+d-1)%d][i2]+x[(i1+1)%d][i2]+x[i1][(i2+d-1)%d]+x[i1][(i2+1)%d])
                    diff_expect+=( -diff_E/(1+np.exp(J_model*diff_E)) )/(N_sample*d**2)
        J_model-=lr*diff_expect
        error=J_model-J_data
        print(abs(J_model-J_data))
