#! /usr/bin/env python
#-*-coding:utf-8-*-
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
np.random.seed(1)
t_interval = 100
#parameter ( System )
d, N_sample,N_model = 6,100,30 #124, 1000
N_remove=100
#parameter ( MPF+GD )
lr,eps =0.01, 1.0e-100
t_gd_max=2000 
def gen_mcmc(J1=[[]],J2=[[]],x=[[]]):
    for i1 in range(d):
        for i2 in range(d):
            #Heat Bath
            diff_E=2.0*x[i1][i2]*(J1[(i1+d-1)%d][i2]*x[(i1+d-1)%d][i2]+J1[i1][i2]*x[(i1+1)%d][i2]
                    +J2[i1][(i2+d-1)%d]*x[i1][(i2+d-1)%d]+J2[i1][i2]*x[i1][(i2+1)%d])#E_new-E_old
            r=1.0/(1+np.exp(diff_E)) 
            R=np.random.uniform(0,1)
            if(R<=r):
                x[i1][i2]=x[i1][i2]*(-1)
    return x

if __name__ == '__main__':
    x=np.random.choice([-1,1],(d,d))
    #Target Structure of the graph
    # vertical parameter
    J1_data=[[0,0,0,0,0,0],
            [0,1,0,0,0,0],
            [0,0,1,1,0,0],
            [0,0,0,1,0,0],
            [0,1,0,1,0,0],
            [0,0,0,0,0,0]]
    # horizontal parameter
    J2_data=[[0,0,0,0,0,0],
            [0,1,0,1,1,0],
            [0,1,0,0,1,1],
            [0,1,1,0,0,0],
            [0,0,0,1,1,0],
            [0,0,0,0,0,0]]
    J1_model=np.ones((d,d))
    J2_model=np.ones((d,d))
    C1_data=np.zeros((d,d))
    C2_data=np.zeros((d,d))
    for n in range(N_sample+N_remove):
        if(n<N_remove):
            x=np.copy(gen_mcmc(J1_data,J2_data,x))
        else:
            for t in range(t_interval):
                x=np.copy(gen_mcmc(J1_data,J2_data,x))
            for i1 in range(d):
                for i2 in range(d):
                    C1_data[i1][i2]+=x[i1][i2]*x[(i1+1)%d][i2]/N_sample
                    C2_data[i1][i2]+=x[i1][i2]*x[i1][(i2+1)%d]/N_sample

    for t_gd in range(t_gd_max):
        xm=np.random.choice([-1,1],(d,d))
        C1_model=np.zeros((d,d))
        C2_model=np.zeros((d,d))
        for m in range(N_model+N_remove):
            if(m<N_remove):
                xm=np.copy(gen_mcmc(J1_model,J2_model,xm))
            else:
                for t in range(t_interval):
                    xm=np.copy(gen_mcmc(J1_model,J2_model,xm))
                for i1 in range(d):
                    for i2 in range(d):
                        C1_model[i1][i2]+=xm[i1][i2]*xm[(i1+1)%d][i2]/N_model
                        C2_model[i1][i2]+=xm[i1][i2]*xm[i1][(i2+1)%d]/N_model

        J1_model=J1_model + lr * (C1_data - C1_model)
        J2_model=J2_model + lr * (C2_data - C2_model)
        
        print(sum(sum(abs(J1_model-J1_data))), sum(sum(abs(J2_model-J2_data))) )

