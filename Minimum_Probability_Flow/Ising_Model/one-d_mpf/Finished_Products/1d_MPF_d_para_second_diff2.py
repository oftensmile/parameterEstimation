#2016/05/19
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
#t_burn_emp, t_burn_model = 1100, 10#10000, 100
t_interval = 40
#parameter ( System )
d, N_sample = 8,300 #124, 1000
N_remove = 100
#parameter ( MPF+GD )
lr,eps =0.1, 1.0e-100
t_gd_max=400 
def gen_mcmc(J=[],x=[] ):
    for i in range(d):
        #Heat Bath
        diff_E=2.0*x[i]*(J[i]*x[(d+1)%d]+J[(i+d-1)%d]*x[(i+d-1)%d])
        r=1.0/(1+np.exp(diff_E)) 
        #r=np.exp(-diff_E) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x

#######    MAIN    ########
#Generate sample-dist
J_max,J_min=1.0,0.0
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
#In this case I applied 
theta_model=np.random.uniform(0,4,d)    #Initial guess
init_theta=np.copy(theta_model)
print("#diff_E diff_E1_nin diff_E2_nin")
error_func=1000
for t_gd in range(t_gd_max):
    error_prev=error_func
    gradK=np.zeros(d)
    n_bach=len(X_sample)
    for nin in range(n_bach):
        x_nin=np.copy(X_sample[nin])
        gradK_nin=np.zeros(d)
        x_nin_shift=np.copy(x_nin[1:d])
        x_nin_shift=np.append(x_nin_shift,x_nin[0])
        x_nin_x_shift=x_nin*x_nin_shift
        E_nin=np.dot(x_nin_x_shift,theta_model)
        for l1 in range(d):
            x_nin_l=np.copy(x_nin)
            x_nin_l[l1]*=-1
            x_nin_l_shift=np.copy(x_nin_l[1:d])
            x_nin_l_shift=np.append(x_nin_l_shift,x_nin_l[0])
            x_nin_l_x_shift=x_nin_l*x_nin_l_shift
            E_nin_l=np.dot(x_nin_l_x_shift,theta_model)
            diff_E=E_nin_l-E_nin
            gradK_nin=gradK_nin-(x_nin_x_shift-x_nin_l_x_shift)*np.exp(0.5*diff_E)/d
            for l2 in range(d):
                if(l2!=l1):
                    x_nin_l_l2=np.copy(x_nin_l)
                    x_nin_l_l2[l2]*=-1
                    x_nin_l_l2_shift=np.copy(x_nin_l_l2[1:d])
                    x_nin_l_l2_shift=np.append(x_nin_l_l2_shift,x_nin_l_l2[0])
                    x_nin_l_l2_x_shift=x_nin_l_l2*x_nin_l_l2_shift
                    E_nin_l_l2=np.dot(x_nin_l_l2_x_shift,theta_model)
                    diff_E_l_l2=E_nin_l_l2-E_nin
                    gradK_nin=gradK_nin-(x_nin_x_shift-x_nin_l_l2_x_shift)*np.exp(0.5*diff_E_l_l2)/(d*(d-1))

        gradK=gradK+gradK_nin/n_bach
    
    
    theta_model=theta_model-lr*gradK
    sum_of_gradK=np.sum(np.sum(gradK))
    error_func=np.sum(np.abs(theta_model-J_vec))/d
    print(t_gd,sum_of_gradK,error_func)
    #if(error_prev<error_func):
    #    break

#Plot
"""
bins=np.arange(1,d+1)
bar_width=0.2
plt.bar(bins,J_vec,color="blue",width=bar_width,label="true",align="center")
plt.bar(bins+bar_width,theta_model,color="red",width=bar_width,label="estimated",align="center")
plt.bar(bins+2*bar_width,init_theta,color="green",width=bar_width,label="initial",align="center")
plt.bar(bins+3*bar_width,gradK*10,color="gray",width=bar_width,label="gradK",align="center")
plt.legend()
filename="test_output_fixed6.png"
plt.savefig(filename)
plt.show()
"""
