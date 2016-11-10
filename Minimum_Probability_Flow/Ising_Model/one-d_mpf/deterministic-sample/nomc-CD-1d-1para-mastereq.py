#2016/08/05
##############
#   H = -J*sum(xixj), J in R^1
##############
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
np.random.seed(1)
#parameter ( MCMC )
t_interval = 40
d, N_sample =16,10 #124, 1000
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
        correlation_data+=calc_C(x_new)/N_sample
        X_sample = x_new
    elif(n>N_remove):
        x_new=np.copy(x)
        #correlation_data+=calc_C(x_new)/N_sample
        X_sample=np.vstack((X_sample,x_new))

J_model=2.0
for t_gd in range(t_gd_max):
    diff_expect=0.0#np.zeros(d)
    for m in range(N_sample):
        x_m=np.copy(X_sample[m])
        for l in range(d):
            diff_E=2*x_m[l]*(x_m[(l+1)%d]+x_m[(l-1+d)%d])
            diff_expect+=( - diff_E * (d*(1+np.exp(J_model*diff_E)))**(-1))/N_sample
    J_model-=lr*diff_expect
    error=np.abs(J_model - J_data)
    print(t_gd,error)
print("#(J_data,J_model)=",J_data,J_model)
