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
d, N_sample =16,2 #124, 1000
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

#######    MAIN    ########
##Generate sample-dist
#J_max,J_min=1.0,0.0
#J_vec=np.random.uniform(J_min,J_max,d)
J_data=1.0
x = np.random.choice([-1,1],d)
correlation_data=0.0#np.zeros(d)
##SAMPLING
for n in range(N_sample+N_remove):
    for t in range(t_interval):
        x = np.copy(gen_mcmc(J_data,x))
    if(n==N_remove):
        x_new=np.copy(x)
        for i in range(d):
            correlation_data+=x_new[i]*x_new[(i+1)%d]/N_sample
        X_sample = x_new
    elif(n>N_remove):
        x_new=np.copy(x)
        X_sample=np.vstack((X_sample,x_new))
        for i in range(d):
            correlation_data+=x_new[i]*x_new[(i+1)%d]/N_sample
print(np.shape(X_sample)) 
#theta_model=np.random.uniform(0,4,d)    #Initial guess
J_model=2.0
for t_gd in range(t_gd_max):
    #MCMC-mean(using CD-method)
    correlation_model=0.0#np.zeros(d)
    for m in range(N_sample):
        x_m=np.copy(X_sample[m])
        for l in range(d):
            J_model+=x_m[l]*x_m[(l+1)%d]/(1 + np.exp(-2.0*x_m[l]*J_model*(x_m[(l+1)%d] + x_m[(l+d-1)%d])))/(N_sample)
    J_model=J_model-(correlation_model-correlation_data)
    #error=np.sqrt(np.sum((theta_model-J_vec)**2))/d
    error=np.abs(J_model - J_data)
    print(t_gd,error )

