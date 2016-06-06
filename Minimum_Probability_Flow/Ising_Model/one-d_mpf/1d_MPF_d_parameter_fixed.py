#2016/05/19
##############
#   H = J*sum(xixj), J in R^1
##############
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
import csv 
np.random.seed(2)
#parameter ( MCMC )
#t_burn_emp, t_burn_model = 1100, 10#10000, 100
t_interval = 10
#parameter ( System )
d, N_sample = 32,1100 #124, 1000
N_remove = 100
#parameter ( MPF+GD )
lr,eps =0.5, 1.0e-100
t_gd_max=700 
def gen_mcmc(J=[],x=[] ):
    for i in range(d):
        #Heat Bath
        diff_E=-2.0*x[i]*(J[i]*x[(d+1)%d]+J[(i+d-1)%d]*x[(i+d-1)%d])
        #r=1.0/(1+np.exp(diff_E)) 
        r=np.exp(-diff_E) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x

#######    MAIN    ########
#Generate sample-dist
J_max,J_min=2.0,0.0
J_vec=np.random.uniform(J_min,J_max,d)
x = np.random.uniform(-1,1,d)
x = np.array(np.sign(x))
#SAMPLING
for n in range(N_sample):
    for t in range(t_interval):
        x = np.copy(gen_mcmc(J_vec,x))
    if(n==N_remove):X_sample = np.copy(x)
    elif(n>N_remove):X_sample=np.vstack((X_sample,np.copy(x)))
#MPF
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
            xl_xl_plu_1=x_nin[l]*x_nin[(l+1)%d]
            xl_min_1_xl=x_nin[(l+d-1)%d]*x_nin[l]
            xl_plu_1_xl_pul_2=x_nin[(l+1)%d]*x_nin[(l+2)%d]
            gradK_nin[l]= - xl_xl_plu_1*np.exp( -  xl_xl_plu_1*theta_model[l] ) *(1.0/d)
            gradK_nin[l]*= ( np.exp(-xl_min_1_xl*theta_model[(l+d-1)%d])+np.exp(-xl_plu_1_xl_pul_2*theta_model[(l+1)%d]) )
        gradK=np.copy(gradK)+gradK_nin*(1.0/n_bach)
    theta_model=theta_model-lr*gradK
    #theta_model=theta_model-lr*(gradK + 0.005*theta_model)
    error_func=np.sum(np.abs(theta_model-J_vec))/d
    print(t_gd,error_func)
#Plot
bins=np.arange(1,d+1)
bar_width=0.2
plt.bar(bins,J_vec,color="blue",width=bar_width,label="true",align="center")
plt.bar(bins+bar_width,theta_model,color="red",width=bar_width,label="estimated",align="center")
plt.bar(bins+2*bar_width,init_theta,color="green",width=bar_width,label="initial",align="center")
plt.bar(bins+3*bar_width,gradK*10,color="gray",width=bar_width,label="gradK",align="center")
plt.legend()
filename="test_output_fixed2.png"
plt.savefig(filename)
plt.show()
