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
d, N_sample =16,300 #124, 1000
num_mcmc_sample=50
N_remove = 100
lr,eps =0.1, 1.0e-100
t_gd_max=200 
def gen_mcmc(J=[],x=[] ):
    for i in range(d):
        #Heat Bath
        diff_E=2.0*x[i]*(J[i]*x[(d+1+i)%d]+J[(i+d-1)%d]*x[(i+d-1)%d])
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
##Generate sample-dist
J_max,J_min=1.0,0.0
J_vec=np.random.uniform(J_min,J_max,d)
x = np.random.choice([-1,1],d)
correlation_data=np.zeros(d)
##SAMPLING
for n in range(N_sample+N_remove):
    for t in range(t_interval):
        x = np.copy(gen_mcmc(J_vec,x))
    if(n==N_remove):
        x_new=np.copy(x)
        for i in range(d):
            correlation_data[i]=x_new[i]*x_new[(i+1)%d]/N_sample
        X_sample = x_new
    elif(n>N_remove):
        x_new=np.copy(x)
        for i in range(d):
            correlation_data[i]+=x_new[i]*x_new[(i+1)%d]/N_sample
        X_sample=np.vstack((X_sample,x_new))
 
theta_model=np.random.uniform(0,4,d)    #Initial guess
for t_gd in range(t_gd_max):
    gradl=np.zeros(d)
    #MCMC-mean(using CD-method)
    correlation_model=np.zeros(d)
    for m in range(num_mcmc_sample):
        x_init=np.copy(X_sample[np.random.randint(N_sample)])
        x_new_for_mcmc=np.copy(gen_mcmc(theta_model,x_init))
        if (m==0):
            for j in range(d):
                correlation_model[j]=x_new_for_mcmc[j]*x_new_for_mcmc[(j+1)%d]/num_mcmc_sample
        elif(m>0):
            for j in range(d):
                correlation_model[j]+=x_new_for_mcmc[j]*x_new_for_mcmc[(j+1)%d]/num_mcmc_sample
    theta_model=theta_model-(correlation_model-correlation_data)
    error=np.sqrt(np.sum((theta_model-J_vec)**2))/d
    print(t_gd,error )

