#2016/08/05
##############
#   H = -J*sum(xixj), J in R^1
##############
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
import csv 
np.random.seed(10)
#parameter ( MCMC )
t_interval = 40
d, N_sample =16,200 #124, 1000
N_remove = 100
lr,eps =0.1, 1.0e-100
t_gd_max=200 
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

def calc_E(x_tot=[[]],theta=[[]]):
    len_x=len(x_tot)
    E=np.zeros(len_x)
    for n in range(len_x):
        x_n=np.copy(x_tot[n])
        E[n]=np.matrix(x_n)*np.matrix(theta)*np.matrix(x_n).T
    return E

#######    MAIN    ########
J_true=0.5
x = np.random.choice([-1,1],d)
correlation_data=np.zeros(d)
##SAMPLING
sum_correlation_data_vec=np.zeros(N_sample)
for n in range(N_sample+N_remove):
    for t in range(t_interval):
        x = np.copy(gen_mcmc(J_true,x))
    if(n==N_remove):
        x_new=np.copy(x)
        for i in range(d):
            correlation_data[i]=x_new[i]*x_new[(i+1)%d]/N_sample
            sum_correlation_data_vec[n-N_remove]+=x_new[i]*x_new[(i+1)%d]
        X_sample = x_new
    elif(n>N_remove):
        x_new=np.copy(x)
        for i in range(d):
            correlation_data[i]+=x_new[i]*x_new[(i+1)%d]/N_sample
            sum_correlation_data_vec[n-N_remove]+=x_new[i]*x_new[(i+1)%d]
        X_sample=np.vstack((X_sample,x_new))
 
######### L(theta)=sum( theta*sum(xixj) - log((2cosh(theta))**d+(2cosh(theta))**d) ) #########
theta_model=2.0
theta_slice=np.arange(-2.0,2.0,0.025)
sum_correlation_data=np.sum(correlation_data)
for th in theta_slice:
    #MCMC-mean(using CD-method)
    CD_of_th=0.0
    for m in range(N_sample):
        CD_of_th_m=th*sum_correlation_data_vec[m]-np.log( (2*np.cosh(th))**d + (2*np.sinh(th))**d)
        p_of_xm=0.0
        xm=np.copy(X_sample[m])
        for j in range(d):
            p_of_xm+=1.0/(1.0 + np.exp(-2.0*th*xm[j]*(xm[(j+1)%d]+xm[(j-1+d)%d]))) / d
        CD_of_th_m*=1.0 + p_of_xm
        CD_of_th+=CD_of_th_m / N_sample
    print(th,CD_of_th)
