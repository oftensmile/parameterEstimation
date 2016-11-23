#! /usr/bin/env python
#-*-coding:utf-8-*-
import numpy as np
import time 
from scipy import linalg
from scipy.optimize import root 
from scipy.optimize import newton 
import math
n_estimation=300
np.random.seed(0)
t_interval = 10
#parameter ( System )
d, N_sample = 4,400 #124, 1000
N_remove=100
#parameter ( MPF+GD )
lr,eps =0.1, 1.0e-100
n_mfa = 50 #Number of the sample for Mean Field Aproximation.
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
    gradK=0.0
    for M in data_matrix:
        x=np.copy(M.mat)
        for i1 in range(d):
            for i2 in range(d):
                diff_E=-2*x[i1][i2]*(x[(i1+d-1)%d][i2]+x[(i1+1)%d][i2]+x[i1][(i2+d-1)%d]+x[i1][(i2+1)%d])
                gradK+=diff_E*np.exp(0.5*J*diff_E)/(d**2)
    return gradK 

def myobj(J,data_matrix):
    J_est=np.zeros(d*d*2)
    for M in data_matrix:
        x=np.copy(M.mat)
        for i1 in range(d):
            for i2 in range(d):
                #i1 direction
                J_est[i1+i2*d]+= x[i1][i2]*x[(i1+1)%d][i2] - x[(i1+1)%d][i2]*np.tanh(
                        x[(i1+1)%d][i2]*J[i1+i2*d]+x[(i1-1+d)%d][i2]*J[(i1-1+d)%d+i2*d]
                       +x[i1][(i2-1+d)%d]*J[d*d+i1+(i2-1+d)%d*d]+x[i1][(i2+1)%d]*J[d*d+i1+i2*d] )
                
                #J_est[i1+i2*d]+= x[(i1+1)%d][i2]*x[(i1+2)%d][i2] - x[(i1+2)%d][i2]*np.tanh(
                #        x[(i1+2)%d][i2]*J[(i1+1)%d+i2*d]+x[i1][i2]*J[i1+i2*d]
                #       +x[(i1+1)%d][(i2-1+d)%d]*J[d*d+(i1+1)%d+(i2-1+d)%d*d]+x[(i1+1)%d][(i2+1)%d]*J[d*d+(i1+1)%d+i2*d] )

                #i2 direction
                J_est[d*d+i1+i2*d]+= x[i1][i2]*x[i1][(i2+1)%d] - x[i1][(i2+1)%d]*np.tanh(
                        x[(i1+1)%d][i2]*J[i1+i2*d]+x[(i1-1+d)%d][i2]*J[(i1-1+d)%d+i2*d]
                       +x[i1][(i2-1+d)%d]*J[d*d+i1+(i2-1+d)%d*d]+x[i1][(i2+1)%d]*J[d*d+i1+i2*d] )
                
                #J_est[d*d+i1+i2*d]+= x[i1][(i2+1)%d]*x[i1][(i2+2)%d] - x[i1][(i2+2)%d]*np.tanh(
                #        x[(i1+1)%d][(i2+1)%d]*J[i1+(i2+1)%d*d]+x[(i1-1+d)%d][(i2+1)%d]*J[(i1-1+d)%d+(i2+1)%d*d]
                #       +x[i1][i2%d]*J[d*d+i1+i2*d]+x[i1][(i2+2)%d]*J[d*d+i1+(i2+1)%d*d] )
    return J_est 

if __name__ == '__main__':
    #sample_list=[500,1000,5000,10000,50000]
    #fname_sample="Pseudo-Likhood.dat"
    #F=open(fname_sample,"w")
    #for N_sample in sample_list:
        #fname="sample"+str(N_sample)+"-PLkhood.dat"
        #f=open(fname,"w")
    J_model_list=np.zeros(n_estimation)
        #for nf in range(n_estimation):
    #J_data=[0,1.0,1.0,0.5,0.0,1.0]
    #SAMPLING-Tmat                      1
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
    
    J_model=root(myobj,0.1*np.ones(d*d*2),args=(data_matrix),method="hybr") 
    print("#J_data=",J_data)
    print("#sol=",J_model.x)
    print("#diff=",np.sum(np.abs(J_data-J_model.x)))

    """
    for i in range(d):
        Ji_newtoon=newton(myobj, 0.1,args=(i,X_sample,))
        J_model_list[i]=Ji_newtoon
    J_newton=np.mean(J_model_list)
    """
            #f.write(str(J_newton)+"  "+str(np.abs(J_newton-J_data))+"\n")
        #f.write("#"+str(N_sample)+"  "+str(np.mean(J_model_list))+"  "+str(np.std(J_model_list))+"\n" )
        #f.close()
        #F.write(str(N_sample)+"  "+str(np.mean(J_model_list))+"  "+str(np.std(J_model_list))+"\n" )
    #F.close()
