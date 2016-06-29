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
d, N_sample = 3,200 #124, 1000
N_remove = 100
#parameter ( MPF+GD )
lr,eps =0.1, 1.0e-100
t_gd_max=1400 
def gen_mcmc(J=[[]],x=[] ):
    for i in range(d):
        diff_E=0
        for j in range(i+1,d):
            #Heat Bath
            diff_E+=2.0*2*x[i]*x[j]*(J[i][j]+J[j][i])#[(d+1)%d]+J[(i+d-1)%d]*x[(i+d-1)%d])
        r=1.0/(1+np.exp(diff_E)) 
        #r=np.exp(-diff_E) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x

#######    MAIN    ########
#Generate sample-dist
J_max,J_min=1.0,0.0
J_mat=np.random.uniform(J_min,J_max,(d,d))
J_mat=0.5*( np.copy(J_mat)+np.copy(J_mat.T) )
for i in range(d):J_mat[i][i]=0
x = np.random.uniform(0,1,d)
x = np.array(np.sign(x))
#SAMPLING
for n in range(N_sample):
    for t in range(t_interval):
        x = np.copy(gen_mcmc(J_mat,x))
    if(n==N_remove):X_sample = np.copy(x)
    elif(n>N_remove):X_sample=np.vstack((X_sample,np.copy(x)))
#MPF
#In this case I applied 
theta_model=np.random.uniform(0,2,(d,d))    #Initial guess
theta_model=0.5*( np.copy(theta_model).T+np.copy(theta_model) )
for i in range(d):theta_model[i][i]=0.0
print("#diff_E diff_E1_nin diff_E2_nin")
for t_gd in range(t_gd_max):
    gradK=np.zeros((d,d))
    n_bach=len(X_sample)
    for nin in range(n_bach):
        x_nin=np.copy(X_sample[nin])
        for k in range(d):
            for l in range(k+1,d):
                t_k_vec=np.copy(theta_model[k])
                t_l_vec=np.copy(theta_model[l])
                x_dot_tkvec=np.dot(x_nin,t_k_vec)/d
                x_dot_tlvec=np.dot(x_nin,t_l_vec)/d
                elemnt_kl=-x_nin[k]*x_nin[l]*( np.exp(-x_nin[k]*2*x_dot_tkvec) + np.exp(-x_nin[k]*2*x_dot_tkvec) )
                #print("elemnt_kl=",elemnt_kl)
                gradK[k][l]+=elemnt_kl/n_bach
                gradK[l][k]+=elemnt_kl/n_bach
    #print(theta_model)
    theta_model=theta_model-lr*gradK
    sum_of_gradK=np.sum(np.sum(gradK))/(d*d)
    error_func=np.sum(np.sum(np.abs(theta_model-J_mat)))/(d*d)
    print(t_gd,sum_of_gradK,error_func)
#Plot
#bins=np.arange(1,d+1)
#bar_width=0.2
#plt.bar(bins,J_vec,color="blue",width=bar_width,label="true",align="center")
#plt.bar(bins+bar_width,theta_model,color="red",width=bar_width,label="estimated",align="center")
#plt.bar(bins+2*bar_width,init_theta,color="green",width=bar_width,label="initial",align="center")
#plt.bar(bins+3*bar_width,gradK*10,color="gray",width=bar_width,label="gradK",align="center")
#plt.legend()
#filename="test_output_fixed6.png"
#plt.savefig(filename)
#plt.show()
