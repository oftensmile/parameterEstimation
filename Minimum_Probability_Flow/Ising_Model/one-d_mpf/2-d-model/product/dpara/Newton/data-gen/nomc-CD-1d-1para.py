#! /usr/bin/env python
#-*-coding:utf-8-*-
import numpy as np
import time 
from scipy import linalg
from scipy.optimize import newton
np.random.seed(0)
n_estimation=300
#parameter ( MCMC )
d, N_sample =16,555550#124, 1000
N_remove = 100
lr,eps =1, 1.0e-100
t_gd_max=100 
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

def myob(J,X_sample=[[]]):
    diff_expect=0
    for m in range(N_sample):
        x_m=np.copy(X_sample[m])
        for l in range(d):
            diff_E=2*x_m[l]*(x_m[(l+1)%d]+x_m[(l-1+d)%d])
            diff_expect+=( - diff_E * (d*(1+np.exp(J*diff_E)))**(-1))/N_sample
    return diff_expect

if __name__ == '__main__':
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
