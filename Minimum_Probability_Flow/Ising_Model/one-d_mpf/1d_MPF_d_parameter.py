#2016/05/19
##############
#   H = J*sum(xixj), J in R^1
##############
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
import csv 
np.random.seed(0)
#parameter ( Model )
T_max=1.2
#Temperature Dependance
#= J^-1=kT/J=T/Tc, Tc=J/k=1
n_T=100
dT=T_max/n_T 

#parameter ( MCMC )
t_burn_emp, t_burn_model = 100, 10#10000, 100
t_interval = 10
#parameter ( System )
d, N_sample = 16,300 #124, 1000
#parameter ( MPF+GD )
lr,eps =0.1, 1.0e-100
#n_mfa = 100 #Number of the sample for Mean Field Aproximation.
t_gd_max=110 
def gen_mcmc(J1,J2,x=[] ):
    for i in range(d):
        #Heat Bath
        diff_E=2.0*x[i]*( J1*(x[(i+d-1)%d]+x[(i+1)%d]) + J2* (x[(i+d-2)%d]+x[(i+2)%d]) )#E_new-E_old
        r=1.0/(1+np.exp(diff_E)) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x

#######    MAIN    ########
#Generate sample-dist
J1,J2=1.2,0.0 # =theta_sample
J_max,J_min=2.0,0
J_vec=np.random.uniform(J_min,J_max,d)
x = np.random.uniform(-1,1,d)
x = np.array(np.sign(x))
for t_burn in range(t_burn_emp):
    x=np.copy(gen_mcmc(J1,J2,x))
#SAMPLING
for n in range(N_sample):
    for t in range(t_interval):
        x = np.copy(gen_mcmc(J1,J2,x))
    if(n==0):X_sample = np.copy(x)
    elif(n>0):X_sample=np.vstack((X_sample,np.copy(x)))

#MPF
#theta_model1,theta_model2=3.0, 2.0  #Initial Guess
theta_model=np.random.uniform(0,4,d)    #Initial guess
init_theta=np.copy(theta_model)
print("#diff_E diff_E1_nin diff_E2_nin")
for t_gd in range(t_gd_max):
    gradK=np.zeros(d)
    n_bach=len(X_sample)
    for nin in range(n_bach):
        x_nin=np.copy(X_sample[nin])
        gradK_nin=np.zeros(d)
        #calc gradK_nin_one_by_one
        for l in range(d):
            diff_del_l_nin=-1.0*x_nin[l]*x_nin[(l+1)%d]
            gradK_nin[l]=2.0*diff_del_l_nin*np.exp(diff_del_l_nin*theta_model[l])
        gradK=np.copy(gradK)+gradK_nin*(1.0/n_bach)
    theta_model=theta_model-lr*gradK
    error_func=np.sum(np.abs(theta_model-J_vec))/d
    print(t_gd,error_func)
bins=np.arange(1,d+1)
bar_width=0.2
plt.bar(bins,J_vec,color="blue",width=bar_width,label="true",align="center")
plt.bar(bins+bar_width,theta_model,color="red",width=bar_width,label="estimated",align="center")
plt.bar(bins+2*bar_width,init_theta,color="green",width=bar_width,label="inital",align="center")
plt.bar(bins+3*bar_width,gradK*100,color="gray",width=bar_width,label="gradK",align="center")
plt.legend()
plt.show()
