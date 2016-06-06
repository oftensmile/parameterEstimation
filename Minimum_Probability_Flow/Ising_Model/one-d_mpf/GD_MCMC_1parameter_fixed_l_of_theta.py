#2016/05/19
##############
#   H = J*sum(xixj), J in R^1
##############
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
import csv
np.random.seed(1)
t_interval = 40
#parameter ( System )
d, N_sample = 16,540 #124, 1000
N_remove=40
#parameter ( MPF+GD )
lr,eps =0.01, 1.0e-100
N_model =70 #Number of the sample for Mean Field Aproximation.
#t_gd_max=1000 

def gen_mcmc(J,x=[] ):
    for i in range(d):
        #Heat Bath
        diff_E=2.0*J*x[i]*(x[(i+d-1)%d]+x[(i+1)%d])#E_new-E_old
        r=1.0/(1+np.exp(diff_E)) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x

def calc_E(J,X=[[]]):
    n_bach=len(X)
    E=0.0
    for n in range(n_bach):
        xn=X[n]
        e=0.0
        for i in range(d):
            e+=xn[i]*xn[(i+1)%d]
        e*=J*(1.0/d)
        E+=e
    E/=n_bach
    return E

def calc_M(X=[[]]):
    n_bach=len(X)
    M=0.0
    for n in range(n_bach):
        xn=X[n]
        M+=np.sum(xn)/d
    M/=n_bach
    return M

def calc_C(X=[[]]):
    n_bach = len(X)
    corre_mean=0.0
    for n in range(n_bach):
        xn=X[n]
        corre=0.0
        for i in range(d):
            corre+=xn[i]*xn[(i+1)%d]/d
        corre_mean+=corre
    corre_mean/=n_bach
    return corre_mean
def calc_sum_of_exp_J(theta,X=[[]]):
    n_sample=len(X)
    sum_exp=0.0
    for xn in X:
        y1=np.copy(xn)
        y2=np.append(y1,y1[0])
        y2=y2[1:len(y2)]
        c=np.dot(y1,y2)
        """
        c=0.0
        for i in range(len(xn)):
            c+=xn[i]*xn[(i+1)%d]
        """
        sum_exp+=np.exp(-c*theta)#/n_sample
    return sum_exp
########    MAIN    ########
J=1.5 # =theta_sample
x = np.random.uniform(-1,1,d)
x = np.array(np.sign(x))
#SAMPLING
for n in range(N_sample):
    for t in range(t_interval):
        x = np.copy(gen_mcmc(J,x))
    if(n==N_remove):X_sample = np.copy(x)
    elif(n>N_remove):X_sample=np.vstack((X_sample,np.copy(x)))

#Generate model-dist
x_model = np.array(np.sign(np.random.uniform(-1,1,d)))
theta_min, theta_max=-2.0,2.0
n_theta=100
d_theta=(theta_max-theta_min)/n_theta
corr_data=calc_C(X_sample)
for n_t in range(n_theta):
    theta=theta_min+n_t*d_theta
    likl_data=-theta*corr_data
    for n_model in range(N_model):
        for t in range(t_interval):
            x_model = np.copy(gen_mcmc(theta,x_model))
        if (n_model==N_remove):X_model = np.copy(x_model)
        elif(n_model>N_remove):X_model = np.vstack((X_model,np.copy(x_model)))
    likl_model=-np.log(calc_sum_of_exp_J(theta,X_model))
    likl=likl_data+likl_model
    print(theta,likl)
