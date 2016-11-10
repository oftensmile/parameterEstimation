#! /usr/bin/env python
#-*-coding:utf-8-*-
import numpy as np
import time 
from scipy import linalg
from scipy.optimize import newton
np.random.seed(0)
n_estimation=300
#parameter ( Model )
T_max=1.2
#Temperature Dependance
#= J^-1=kT/J=T/Tc, Tc=J/k=1
n_T=100
dT=T_max/n_T 
J=1.0
#parameter ( MCMC )
#t_burn_emp, t_burn_model = 1000, 10#10000, 100
#parameter ( System )
d, N_sample = 16,100 #124, 1000
N_remove = 100
#parameter ( MPF+GD )
lr,eps =0.01, 1.0e-100
#n_mfa = 100 #Number of the sample for Mean Field Aproximation.
t_gd_max=50 
def gen_mcmc(J,x=[] ):
    for i in range(d):
        #Heat Bath
        diff_E=2.0*J*x[i]*(x[(i+d-1)%d]+x[(i+1)%d])#E_new-E_old
        r=1.0/(1+np.exp(diff_E)) 
        #r=np.exp(-diff_E) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x

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

def myob(J,eta,X_sample=[[]]):
    gradK=0
    for sample in X_sample:
        #x_nin=np.reshep(np.copy(sample),(d,d)
        x_nin=np.copy(sample)
        gradK_nin=0.0
        #hamming distance = 1
        for hd in range(d):
            diff_delE_nin=-2.0*x_nin[hd]*(x_nin[(hd+d-1)%d]+x_nin[(hd+1)%d])
            diff_E_nin=diff_delE_nin*J
            abs_diffE=abs(diff_E_nin)
            if(abs_diffE>0):
                gradK_nin+=(1.0/abs_diffE**eta)*diff_delE_nin*np.exp(0.5*diff_E_nin)/d
        gradK+=gradK_nin/N_sample
    return gradK 
    
if __name__ == '__main__':
    #eta_list=[-1,-0.5,0.0,0.4,0.5,0.6]
    eta_list=[-1,-0.5]
    for eta in eta_list:
        fname_sample="MPF-eta"+str(eta)+".dat"
        F=open(fname_sample,"w")
        sample_list=[100]
        for N_sample in sample_list:
            fname="sample"+str(N_sample)+"-MPF-eta"+str(eta)+".dat"
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
                J_newton=newton(myob,0.5,args=(eta,X_sample,))
                J_model_list[nf]=J_newton
                f.write(str(J_newton)+"  "+str(np.abs(J_newton-J_data))+"\n")
            f.write("#"+str(N_sample)+"  "+str(np.mean(J_model_list))+"  "+str(np.std(J_model_list))+"\n" )
            f.close()
            F.write(str(N_sample)+"  "+str(np.mean(J_model_list))+"  "+str(np.std(J_model_list))+"\n" )
        F.close()

