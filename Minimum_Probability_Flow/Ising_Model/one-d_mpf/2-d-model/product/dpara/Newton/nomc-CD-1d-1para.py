#! /usr/bin/env python
#-*-coding:utf-8-*-
import numpy as np
import time 
from scipy import linalg
from scipy.optimize import newton
from scipy.optimize import root
np.random.seed(0)
n_estimation=300
#parameter ( MCMC )
d, N_sample =6,50#124, 1000
N_remove = 100
lr,eps =1, 1.0e-100
t_gd_max=100
t_interval=20
class Matrices():
    def __init__(self,mat):
        self.mat=mat

def build_matrix(x=[[]]):
    global matrices
    matrices=tuple(Matrices(x) for i in range(1))
 
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

def calc_C(x=[]):
    corre=0.0
    for i in range(d):
        corre+=x[i]*x[(i+1)%d]
    return corre

def myob(J1=[[]],J2=[[]]):
    diff_expect1=np.zeros((d,d))
    diff_expect2=np.zeros((d,d))
    #Is it possible to use data_matrix without define?
    for M in data_matrix:
        x=np.copy(M.mat)
        for i1 in range(d):
            for i2 in range(d):
                diff_E=2.0*x[i1][i2]*(J1_model[(i1+d-1)%d][i2]*x[(i1+d-1)%d][i2]+J1_model[i1][i2]*x[(i1+1)%d][i2]
                +J2_model[i1][(i2+d-1)%d]*x[i1][(i2+d-1)%d]+J2_model[i1][i2]*x[i1][(i2+1)%d])#E_new-E_old
                diff_expect1[i1][i2]+=(-2.0*x[i1][i2]*x[(i1+1)%d][i2])/((1+np.exp(diff_E))*N_sample*d**2)
                diff_expect1[(i1+d-1)%d][i2]+=(-2.0*x[i1][i2]*x[(i1+d-1)%d][i2])/((1+np.exp(diff_E))*N_sample*d**2)
                diff_expect2[i1][i2]+=(-2.0*x[i1][i2]*x[i1][(i2+1)%d])/((1+np.exp(diff_E))*N_sample*d**2)
                diff_expect2[i1][(i2+d-1)%d]+=(-2.0*x[i1][i2]*x[i1][(i2+d-1)%d])/((1+np.exp(diff_E))*N_sample*d**2)
    return [diff_expect1,diff_expect2] 
    
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
    #C1_data=np.zeros((d,d))
    #C2_data=np.zeros((d,d))
    for n in range(N_sample+N_remove):
        if(n<N_remove):
            x=np.copy(gen_mcmc(J1_data,J2_data,x))
        elif(n==N_remove):
            x=np.copy(gen_mcmc(J1_data,J2_data,x))
            build_matrix(x)
            data_matrix=matrices
        else:
            for t in range(t_interval):
                x=np.copy(gen_mcmc(J1_data,J2_data,x))
            data_matrix=np.append(data_matrix,Matrices(x))
    print("sucess")
    J1_init=np.random.random((d,d))
    J2_init=np.random.random((d,d))
    J1sol,J2sol=root(myob,J1_init,J2_init)


"""
    sample_list=[50,100,500,1000,5000,10000]
    fname_sample="CD1_nomc.dat"
    F=open(fname_sample,"w")
    for N_sample in sample_list:
        fname="sample"+str(N_sample)+"-nomcCD1.dat"
        f=open(fname,"w")
        J_model_list=np.zeros(n_estimation)
        for nf in range(n_estimation): 
            J_data=1.0
            #SAMPLING-Tmat
            for n in range(N_sample):
                x=get_sample(J_data)
                if(n==0):
                    X_sample = np.copy(x)
                elif(n>0):
                    X_sample=np.vstack((X_sample,np.copy(x)))
    #J_model=2.0
            J_newton=newton(myob,0.5,args=(X_sample,))
            #print("nf=",nf,",  J=",J_newton)
            J_model_list[nf]=J_newton
            f.write(str(J_newton)+"  "+str(np.abs(J_newton-J_data))+"\n")
        f.write("#"+str(N_sample)+"  "+str(np.mean(J_model_list))+"  "+str(np.std(J_model_list))+"\n" )
        f.close()
        F.write(str(N_sample)+"  "+str(np.mean(J_model_list))+"  "+str(np.std(J_model_list))+"\n" )
    F.close()

    """
