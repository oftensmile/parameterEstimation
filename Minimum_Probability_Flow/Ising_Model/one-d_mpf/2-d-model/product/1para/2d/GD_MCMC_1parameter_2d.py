#! /usr/bin/env python
#-*-coding:utf-8-*-
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
np.random.seed(1)
t_interval = 100
#parameter ( System )
d, N_sample,N_model = 8,100,30 #124, 1000
N_remove=100
#parameter ( MPF+GD )
lr,eps =0.001, 1.0e-100
t_gd_max=1000 
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

def calc_C(x=[[]]):
    c=0.0
    for i1 in range(d):
        for i2 in range(d):
            c+=x[i1][i2]*(x[(i1+d-1)%d][i2]+x[(i1+1)%d][i2]+x[i1][(i2+d-1)%d]+x[i1][(i2+1)%d])
    return 0.5*c

if __name__ == '__main__':
    x=np.random.choice([-1,1],(d,d))
    J_data,J_model=1.0,2.0
    C_data,C_model=0.0,0.0
    for n in range(N_sample+N_remove):
        if(n<N_remove):
            x=np.copy(gen_mcmc(J_data,x))
        else:
            for t in range(t_interval):
                x=np.copy(gen_mcmc(J_data,x))
            C_data+=calc_C(x)/N_sample
    
    for t_gd in range(t_gd_max):
        xm=np.random.choice([-1,1],(d,d))
        for m in range(N_model+N_remove):
            if(m<N_remove):
                xm=np.copy(gen_mcmc(J_model,xm))
            else:
                for t in range(t_interval):
                    xm=np.copy(gen_mcmc(J_model,xm))
                C_model+=calc_C(xm)/N_model
        J_model=J_model - lr * (C_data - C_model)
        C_model=0.0
        print(t_gd,abs(J_model-J_data))

