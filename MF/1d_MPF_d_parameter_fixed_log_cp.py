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
d, N_sample = 32,300 #124, 1000
N_remove = 100
#parameter ( MPF+GD )
lr,eps =0.1, 1.0e-100
t_gd_max=2000 
def gen_mcmc(J=[[]],x=[] ):
    for i in range(d):
        diff_E=0
        for j in range(d):
        #Heat Bat
            diff_E+=2.0*x[i]*x[j]*(J[i][j]+J[j][i])
        r=1.0/(1+np.exp(diff_E)) 
        #r=np.exp(-diff_E) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x

#######    MAIN    ########
#Generate sample-dist
J_max,J_min=0.01,0.0
J_vec=np.random.uniform(J_min,J_max,d)
J_mat=np.zeros((d,d))
for l in range(d):
    J_mat[l][(l+1)%d]=J_vec[l]
    J_mat[(l+1)%d][l]=J_vec[l]

x = np.random.uniform(-1,1,d)
x = np.array(np.sign(x))
#SAMPLING
for n in range(N_sample+N_remove):
    for t in range(t_interval):
        x = np.copy(gen_mcmc(J_mat,x))
    if(n==N_remove):X_sample = np.copy(x)
    elif(n>N_remove):X_sample=np.vstack((X_sample,np.copy(x)))
#MPF
#In this case I applied 
theta_model=np.random.uniform(0,1,d)    #Initial guess
init_theta=np.copy(theta_model)
time_i=time.time()
for t_gd in range(t_gd_max):
    gradK=np.zeros(d)
    n_bach=len(X_sample)
    for nin in range(n_bach):
        x_nin=np.copy(X_sample[nin])
        gradK_nin=np.zeros(d)
        for l in range(d):
            xl_xl_plu_1=x_nin[l]*x_nin[(l+1)%d]
            xl_min_1_xl=x_nin[(l+d-1)%d]*x_nin[l]
            xl_plu_1_xl_pul_2=x_nin[(l+1)%d]*x_nin[(l+2)%d]
            gradK_nin[l]= -xl_xl_plu_1*np.exp( -xl_xl_plu_1*theta_model[l] ) /d
            gradK_nin[l]*= ( np.exp(-xl_min_1_xl*theta_model[(l+d-1)%d])+np.exp(-xl_plu_1_xl_pul_2*theta_model[(l+1)%d]) )
            gradK[l]+=gradK_nin[l]/n_bach
    
    theta_model=theta_model-lr*gradK
    sum_of_gradK=np.sum(gradK)
    error_func=np.sum(np.abs(theta_model-J_vec))/d
    print(t_gd,sum_of_gradK,error_func)
theta_model_mat=np.zeros((d,d))
for l in range(d):
    theta_model_mat[l][(l+1)%d]=theta_model[l]
    theta_model_mat[(l+1)%d][l]=theta_model[l]
#Plot
time_f=time.time()
dtime=time_f-time_i
print("calc time =",dtime)
plt.figure()
plt.subplot(131)
plt.imshow(J_mat)
plt.title("Jtrue")
plt.colorbar()
plt.subplot(132)
plt.imshow(theta_model_mat)
plt.title("Jest")
plt.colorbar()
plt.subplot(133)
plt.imshow(J_mat-theta_model_mat)
plt.title("Jtrue-Jest")
plt.colorbar()
plt.show()
