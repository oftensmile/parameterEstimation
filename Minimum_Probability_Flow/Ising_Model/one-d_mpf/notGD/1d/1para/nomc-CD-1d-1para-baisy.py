#! /usr/bin/env python
#-*-coding:utf-8-*-
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.optimize import newton
np.random.seed(0)
n_estimation=4
#parameter ( MCMC )
d, N_sample =16,10#124, 1000
N_remove = 40
lr,eps =1, 1.0e-100
t_gd_max=500 
def gen_mcmc(J,x=[] ):
    for i in range(d):
        #Heat Bath
        diff_E=2.0*x[i]*J*(x[(d+1+i)%d]+x[(i+d-1)%d])
        r=1.0/(1+np.exp(diff_E)) 
        #r=np.exp(-diff_E) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x

def calc_C(x=[]):
    corre=0.0
    for i in range(d):
        corre+=x[i]*x[(i+1)%d]
    return corre

def Tk(J,k):
    l1=(2*np.cosh(J))**k
    l2=(2*np.sinh(J))**k
    return ( 0.5*(l1+l2) , 0.5*(l1-l2) )

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

def myob(J_model,N_sample,X_sample=[[]]):
    diff_expect=0
    for sample in X_sample:
        #x_nin=np.reshep(np.copy(sample),(d,d)
        x_m=np.copy(sample)
        for l in range(d):
            diff_E=2*x_m[l]*(x_m[(l+1)%d]+x_m[(l-1+d)%d])
            diff_expect+=( - diff_E * (d*(1+np.exp(J_model*diff_E)))**(-1))/N_sample
    return diff_expect
 
if __name__ == '__main__':
    n_estimation=200
    N_sample=[10,20,40,100,200,400,800,1600,3200]
    fname="est"+str(n_estimation)+"-nomcCD.dat"
    f=open(fname,"w")
    J_data=1.0
    for N_sample in N_list:
        error_list=np.zeros(n_estimation)
        for nf in range(n_estimation):
            ##Generate sample-dist
            correlation_data=0.0
            #SAMPLING-Tmat
            x=np.random.choice([-1,1],d)
            for n in range(N_sample+N_remove):
                for t in range(t_gd_max):
                    x=gen_mcmc(J_data,np.copy(x))
                if(n==N_remove):
                    x_new=np.copy(x)
                    X_sample = np.copy(x)
                    correlation_data+=calc_C(x_new)/N_sample
                elif(n>N_remove):
                    x_new=np.copy(x)
                    X_sample=np.vstack((X_sample,np.copy(x)))
                    correlation_data+=calc_C(x_new)/N_sample
            J_model_newton=newton(myob,0.5,args=(N_sample,X_sample,))
            error_list[nf]=J_model_newton-J_data
        f.write(str(N_sample)+"  "+str(np.mean(error_list))+"  "+str(np.std(error_list))+"\n" )
    f.close()
