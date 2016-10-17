#2016/08/05
##############
#   H = -J*sum(xixj), J in R^1
##############
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
import csv 
np.random.seed(0)
#parameter ( MCMC )
d, N_sample =16,200 #124, 1000
num_mcmc_sample=50
N_remove = 200
lr,eps =0.1, 1.0e-100
t_gd_max=200 
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


#######    MAIN    ########
##Generate sample-dist
#J_max,J_min=1.0,0.0
#J_vec=np.random.uniform(J_min,J_max,d)
J_data=1.0
correlation_data=0.0#np.zeros(d)
""" 
#SAMPLING-Simple
x = np.random.choice([-1,1],d)
for n in range(N_sample+N_remove):
    x = np.copy(gen_mcmc(J_data,x))
    if(n==N_remove):
        x_new=np.copy(x)
        X_sample = x_new
        correlation_data+=calc_C(x_new)/N_sample
    elif(n>N_remove):
        x_new=np.copy(x)
        X_sample=np.vstack((X_sample,x_new))
        correlation_data+=calc_C(x_new)/N_sample
""" 
#SAMPLING-Tmat
for n in range(N_sample):
    x=get_sample(J_data)
    if(n==0):
        x_new=np.copy(x)
        X_sample = np.copy(x)
        correlation_data+=calc_C(x_new)/N_sample
    elif(n>0):
        x_new=np.copy(x)
        X_sample=np.vstack((X_sample,np.copy(x)))
        correlation_data+=calc_C(x_new)/N_sample


#theta_model=np.random.uniform(0,4,d)    #Initial guess
J_model=2.0
for t_gd in range(t_gd_max):
    gradl=np.zeros(d)
    #MCMC-mean(using CD-method)
    correlation_model=0.0
    for m in range(N_sample):
        #Using all samples
        #/*THIS CHICE IS VERY IMPORRTANT!! MAYBE*/#
        #x_init=np.copy(X_sample[m])
        x_init=np.copy(X_sample[(np.random.randint(N_sample))])
        x_new_for_mcmc=np.copy(gen_mcmc(J_model,x_init))#This update is possible to generate any state.
        correlation_model+=calc_C(x_new_for_mcmc)/N_sample
    J_model-=lr*(correlation_model-correlation_data)
    #error=np.sqrt(np.sum((theta_model-J_vec)**2))/d
    error=np.abs(J_model-J_data)
    print(t_gd,error)
