#! /usr/bin/env python
#-*-coding:utf-8-*-
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
import csv
from scipy.optimize import fsolve
from scipy.optimize import newton 
from scipy.optimize import minimize 
from scipy.optimize import bisect 
import math
n_estimation=3
np.random.seed(1)
t_interval = 30
#parameter ( System )
d, N_sample = 8,50 #124, 1000
N_remove=40
#parameter ( MPF+GD )
lr,eps =0.1, 1.0e-100
n_mfa = 50 #Number of the sample for Mean Field Aproximation.
t_gd_max=80 
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
            #corre+=xn[i]*xn[(i+1)%d]/d
            corre+=xn[i]*xn[(i+1)%d]
        corre_mean+=corre
    corre_mean/=n_bach
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
    #return -g_data_sum + d*((2*np.cosh(J))**(d-1)+(2*np.sinh(J))**(d-1))/((2*np.cosh(J))**d+(2*np.sinh(J))**d)
    return -g_data_sum + d*((2*np.cosh(J))**(d-1)+(2*np.sinh(J))**(d-1))/((2*np.cosh(J))**d+(2*np.sinh(J))**d)

if __name__ == '__main__':
    #fname="sample"+str(N_sample)+"MCMC.dat"
    #f=open(fname,"w")
    #for nf in range(n_estimation):
    J_data=1.0 # =theta_sample
    #SAMPLING-Tmat
    for n in range(N_sample+N_remove):
        x=get_sample(J_data)
        if(n==N_remove):X_sample = np.copy(x)
        elif(n>N_remove):X_sample=np.vstack((X_sample,np.copy(x)))
    corre_sample_mean=calc_C(X_sample) 
    J_newton = fsolve(Obfunc_1d_1para,0.1,args=(corre_sample_mean))
    #J_bisect = bisect(Obfunc_1d_1para,a=0.0,b=2.0,args=(corre_sample_mean))
    #J_nelder_mead = minimize(Obfunc_1d_1para,0.1,method="Nelder-Mead",args=(corre_sample_mean))
    #J_powell = minimize(Obfunc_1d_1para, 0.1,method="Powell",args=(corre_sample_mean))
    #J_cg = minimize(Obfunc_1d_1para, 0.1,method="CG",args=(corre_sample_mean))
    #J_bfgs = minimize(Obfunc_1d_1para, 0.1,method="BFGS",args=(corre_sample_mean))
   
    xi = np.array(np.sign(np.random.uniform(-1,1,d)))
    theta_model=2.0   #Initial Guess
    t_gd_max=2
    for t_gd in range(t_gd_max):
        for n_model in range(n_mfa+N_remove):
            for t in range(t_interval):
                xi = np.copy(gen_mcmc(theta_model,xi))
            if (n_model==N_remove):Xi_model = np.copy(xi)
            elif(n_model>N_remove):Xi_model = np.vstack((Xi_model,np.copy(xi)))
        corre_model_mean=calc_C(Xi_model)
        grad_likelihood=-corre_sample_mean+corre_model_mean
        theta_model=np.copy(theta_model)-lr*grad_likelihood
        theta_diff = theta_model-J_data
        print(theta_diff)
    print("#J_data= \n",J_data)
    print("#J_model= \n",theta_model)
    print("#J_newton= \n",J_newton)
    #print("#J_bisect= \n",J_bisect)
    #print("#J_nelder_mead= \n",J_nelder_mead)
    #print("#J_powell= \n",J_powell)
    #print("#J_cg= \n",J_cg)
    #print("#J_bfgs= \n",J_bfgs)
    #f.close()
