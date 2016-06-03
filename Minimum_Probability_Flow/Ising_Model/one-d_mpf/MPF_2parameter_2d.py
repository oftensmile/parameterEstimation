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
#t_burn_emp, t_burn_model = 1000, 10#10000, 100
t_interval = 60
#parameter ( System )
d, N_sample = 4,150 #124, 1000
N_remove = 100
#parameter ( MPF+GD )
lr,eps =0.01, 1.0e-100
#n_mfa = 100 #Number of the sample for Mean Field Aproximation.
t_gd_max=40 
def gen_mcmc(J1,J2,x=[[]] ):
    for i in range(d):
        for j in range(d):
            #Heat Bath
            diff_E=2.0*J1*x[i][j]*(x[(i+d-1)%d][j]+x[(i+1)%d][j]) + 2.0*J2*x[i][j]*(x[i][(j+d-1)%d]+x[i][(j+1)%d])# E_new-E_old
            #r=1.0/(1+np.exp(diff_E)) 
            r=1#np.exp(-diff_E) 
            R=np.random.uniform(0,1)
            if(R<=r):
                x[i][j]=x[i][j]*(-1)
    return x

#######    MAIN    ########
#Generate sample-dist
J1,J2=0.0,0.0 # =theta_sample
x = np.random.uniform(-1,1,(d,d))
x = np.array(np.sign(x))
#SAMPLIN

n_J=100
for j1 in range(
for n in range(N_sample):
    for t in range(t_interval):
        x = np.copy(gen_mcmc(J1,J2,x))
        if(n==N_remove):X_sample = np.reshape(np.copy(x),len(x)*len(x.T))
        elif(n>N_remove):X_sample=np.vstack((X_sample,np.reshape(np.copy(x),len(x)*len(x.T))))
n_bach=len(X_sample)
theta_model1,theta_model2=1.5,1.5   #Initial Guess
print("X_sample=\n",X_sample)

mean_magnet=0.0
for n in range(n_bach):
    x_nin=np.copy(X_sample[n])
    print(x_nin)
    sum_magnet=np.sum(x_nin)/(len(x_nin))
    mean_magnet+=sum_magnet/n_bach
print("magnetization=",mean_magnet)



"""
print("#gd-step, abs-grad_likelihood, theta-error")
for t_gd in range(t_gd_max):
    #calc gradK of theta
    gradK1,gradK2=0.0,0.0
    for sample in X_sample:
        x_nin=np.reshape(np.copy(sample),(d,d))
        gradK_nin1,gradK_nin2=0.0,0.0
        #hamming distance = 1
        for ni in range(d):
            for nj in range(d):
                diff_delE_nin1=-2.0*x_nin[ni][nj]*(x_nin[(ni+d-1)%d][nj]+x_nin[(ni+1)%d][nj])
                diff_delE_nin2=-2.0*x_nin[ni][nj]*(x_nin[ni][(nj+d-1)%d]+x_nin[ni][(nj+1)%d])
                diff_E_nin=diff_delE_nin1*theta_model1 + diff_delE_nin2*theta_model2
                gradK_nin1+=diff_delE_nin1*np.exp(0.5*diff_E_nin)
                gradK_nin2+=diff_delE_nin2*np.exp(0.5*diff_E_nin)
        gradK1+=gradK_nin1*(1.0/n_bach)
        gradK2+=gradK_nin2*(1.0/n_bach)
    theta_model1=np.copy(theta_model1) - lr * gradK1
    theta_model2=np.copy(theta_model2) - lr * gradK2
    theta_diff1,theta_diff2=abs(theta_model1-J1),abs(theta_model2-J2)
    print(t_gd,abs(gradK1),abs(gradK2),theta_diff1,theta_diff2)
print("#J1true,J2true=",J1,J2,"the1,the2=",theta_model1,theta_model2)

"""
