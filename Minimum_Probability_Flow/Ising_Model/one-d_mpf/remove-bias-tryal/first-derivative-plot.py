#! /usr/bin/env python
#-*-coding:utf-8-*-
import numpy as np
import time 
from scipy import linalg
from scipy.optimize import root
from scipy.optimize import fsolve
np.random.seed(1)
#parameter ( MCMC )
n_estimation=10
d, N_sample =16,400 #124, 1000
num_mcmc_sample=50
N_remove = 100
lr,eps =0.01, 1.0e-100
t_gd_max=500 
def gen_mcmc(J,x=[]):
    for i in range(d):
        #Heat Bath
        diff_E=2.0*x[i]*J*(x[(d+1+i)%d]+x[(i+d-1)%d])
        r=1.0/(1+np.exp(diff_E)) 
        #r=np.exp(-diff_E) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x

def gen_mcmc_single(J,x=[]):
    index=np.random.randint(d)
    #Heat Bath
    diff_E=2.0*x[index]*J*(x[(d+1+index)%d]+x[(index+d-1)%d])
    r=1.0/(1+np.exp(diff_E)) 
    #r=np.exp(-diff_E) 
    R=np.random.uniform(0,1)
    if(R<=r):
        x[index]=x[index]*(-1)
    return x

def calc_C(x=[]):
    corre=0.0
    for i in range(d):
        corre+=x[i]*x[(i+1)%d]
    return corre

def calc_C_tot(X=[[]]):
    n_bach = len(X)
    corre_mean=0.0
    for n in range(n_bach):
        xn=X[n]
        corre=0.0
        for i in range(d):
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
    return -g_data_sum+(d*np.cosh(J)*np.sinh(J)*(np.cosh(J)**(-2+d)+np.sinh(J)**(-2+d)))/(np.cosh(J)**d+np.sinh(J)**d)


def grad_obj(J,correlation_data,X_sample):
    correlation=0
    for m in range(N_sample):
        x_init=np.copy(X_sample[m])
        #x_init=np.copy(X_sample[(np.random.randint(N_sample))])
        x_new_for_mcmc=np.copy(gen_mcmc(J,x_init))#This update is possible to generate any state.
        correlation+=calc_C(x_new_for_mcmc)/N_sample
    return correlation-correlation_data

if __name__ == '__main__':
    dJ=0.05
    J_list=np.arange(0.0,3.0,dJ)
    #J_list=np.arange(1.0,1.0+dJ*3,dJ)
    J_len=len(J_list)
    mle_list=np.zeros(J_len)
    cd1_list=np.zeros(J_len)
    Dmle_lis=np.zeros(J_len)
    Dcd1_lis=np.zeros(J_len)
    fname="plot-cd-mle-sample"+str(N_sample)+"-4.dat"
    f=open(fname,"w")
    #SAMPLING-Tmat
    J_data=1.0
    n_tryal=1000
    for nt in range(n_tryal):
        np.random.seed(nt)
        
        for n in range(N_sample):
            x=get_sample(J_data)
            if(n==0):X_sample = np.copy(x)
            elif(n>0):X_sample=np.vstack((X_sample,np.copy(x)))
        corre_data=calc_C_tot(X_sample)
        
        nJ=0
        for J in J_list:
            correlation_data=0.0#np.zeros(d)
            del_L = Obfunc_1d_1para(J,corre_data) 
            #J_mle = fsolve(Obfunc_1d_1para,0.1,args=(corre_data))
            del_CD1= grad_obj(J,corre_data,X_sample)
            mle_list[nJ]+=del_L/n_tryal 
            cd1_list[nJ]+=del_CD1/n_tryal
            #print(nJ,", J_mle=",mle_list[nJ])
            #print(nJ,", J_cd=",cd1_list[nJ])
            #print(nJ,", DJ_mle=",Dmle_lis[nJ])
            #print(nJ,", DJ_cd1=",Dcd1_lis[nJ])
            nJ+=1
    nJ=0
    for J in J_list:
        if(nJ>0):
            Dmle_lis[nJ]=(mle_list[nJ]-mle_list[nJ-1])/dJ
            Dcd1_lis[nJ]=(cd1_list[nJ]-cd1_list[nJ-1])/dJ
        f.write(str(J)+"  "+str(mle_list[nJ])+" "+str(cd1_list[nJ])+"  "+str(Dmle_lis[nJ])+" "+ str(Dcd1_lis[nJ])+"\n" )
        nJ+=1
    f.close()
