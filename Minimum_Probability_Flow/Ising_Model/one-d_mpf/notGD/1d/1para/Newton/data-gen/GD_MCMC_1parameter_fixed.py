#! /usr/bin/env python
#-*-coding:utf-8-*-
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.optimize import newton 
import math
n_estimation=300
np.random.seed(0)
t_interval = 10
#parameter ( System )
d, N_sample = 16,50 #124, 1000
N_remove=30
#parameter ( MPF+GD )
n_mfa = 50 #Number of the sample for Mean Field Aproximation.
t_gd_max=100 

def gen_mcmc(J,x=[] ):
    for i in range(d):
        #Heat Bath
        diff_E=2.0*J*x[i]*(x[(i+d-1)%d]+x[(i+1)%d])#E_new-E_old
        r=1.0/(1+np.exp(diff_E)) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x

def calc_C(X=[[]]):
    n_bach = len(X)
    corre_mean=0.0
    for n in range(n_bach):
        xn=X[n]
        corre=0.0
        for i in range(d):
            corre+=xn[i]*xn[(i+1)%d]
        corre_mean+=corre/n_bach
    return corre_mean

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

def Obfunc_1d_1para(J,g_data_sum):
    return -g_data_sum+(d*np.cosh(J)*np.sinh(J)*(np.cosh(J)**(-2+d)+np.sinh(J)**(-2+d)))/(np.cosh(J)**d+np.sinh(J)**d)

if __name__ == '__main__':
    sample_list=[5,10,100,500,1000,5000,10000]
    fname_sample="Likelihood_mcmc.dat"
    F=open(fname_sample,"w")
    for N_sample in sample_list:
        fname="sample"+str(N_sample)+"-lkhd_mcmc.dat"
        f=open(fname,"w")
        J_model_list=np.zeros(n_estimation)
        for nf in range(n_estimation):
            J_data=1.0 # =theta_sample
            #SAMPLING-Tmat
            c_mean_data=0.0
            for n in range(N_sample):
                x=get_sample(J_data)
                if(n==0):
                    X_sample = np.copy(x)
                elif(n>0):
                    X_sample=np.vstack((X_sample,np.copy(x)))
            corre_sample_mean=calc_C(X_sample)
            J_newton=newton(Obfunc_1d_1para,0.5,args=(corre_sample_mean,))
            #print("nf=",nf,",  J=",J_newton)
            J_model_list[nf]=J_newton
            f.write(str(J_newton)+"  "+str(np.abs(J_newton-J_data))+"\n")
        f.write("#"+str(N_sample)+"  "+str(np.mean(J_model_list))+"  "+str(np.std(J_model_list))+"\n" )
        f.close()
        F.write(str(N_sample)+"  "+str(np.mean(J_model_list))+"  "+str(np.std(J_model_list))+"\n" )
    F.close()
