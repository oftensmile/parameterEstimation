#2016/05/19
##############
#   H = J*sum(xixj), J in R^1
##############
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
import csv
np.random.seed(1)
#parameter ( Model )
T_max=1.2
#Temperature Dependance
#= J^-1=kT/J=T/Tc, Tc=J/k=1
n_T=100
dT=T_max/n_T 

#parameter ( MCMC )
t_burn_emp, t_burn_model = 1000, 10#10000, 100
t_interval = 10
#parameter ( System )
d, N_sample = 16,100 #124, 1000
#parameter ( MPF+GD )
lr,eps =0.01, 1.0e-100
n_mfa = 100 #Number of the sample for Mean Field Aproximation.
t_gd_max=15 
def gen_mcmc(J,x=[] ):
    for i in range(d):
        #Heat Bath
        diff_E=2.0*J*x[i]*(x[(i+d-1)%d]+x[(i+1)%d])#E_new-E_old
        r=1.0/(1+np.exp(diff_E)) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x
########    MAIN    ########
#Generate sample-dist
J=1.2 # =theta_sample
x = np.random.uniform(-1,1,d)
x = np.array(np.sign(x))
for t_burn in range(t_burn_emp):
    x=np.copy(gen_mcmc(J,x))
#SAMPLING
for n in range(N_sample):
    for t in range(t_interval):
        x = np.copy(gen_mcmc(J,x))
    if(n==0):X_sample = np.copy(x)
    elif(n>0):X_sample=np.vstack((X_sample,np.copy(x)))
#Jack Knife
n_removed_subset=10
n_sample_tot = len(X_sample)
n_set_of_Jknife=int(n_sample_tot/n_removed_subset)
theta_vec=0.0
for ns in range(n_set_of_Jknife):
    idx_subset_strt,idx_subset_last=ns*n_removed_subset, (ns+1)*n_removed_subset
    set_front=np.arange(0,idx_subset_strt)
    set_back=np.arange(idx_subset_last,n_sample_tot)
    set_Jknife=np.append(set_front,set_back)
    n_bach=len(set_Jknife)
    theta_model=2.0 #Initial Guess
    for t in range(t_gd_max):
        gradK=0.0
        for nin in set_Jknife:
            x_nin=X_sample[nin]
            gradK_nin=0.0
            #hamming distance = 1
            for hd in range(d):
                diff_delE_nin=-2.0*x_nin[hd]*(x_nin[(hd+d-1)%d]+x_nin[(hd+1)%d])
                diff_E_nin=diff_delE_nin*theta_model
                gradK_nin+=diff_delE_nin*np.exp(0.5*diff_E_nin)
            gradK+=gradK_nin
        gradK*=(1.0/n_bach)
    theta_model=np.copy(theta_model) - lr * gradK
    #theta_diff=abs(theta_model-J)
    if(ns==0):theta_vec=theta_model
    elif(ns>=1):theta_vec=np.append(np.copy(theta_vec),theta_model)
    print(theta_model,"#=theta_est, theta_true=",J)

#Simple calc
n_bach=len(X_sample)
for t_gd in range(t_gd_max):
    #calc gradK of theta
    gradK=0.0
    for nin in range(n_bach):
        x_nin=X_sample[nin]
        gradK_nin=0.0
        #hamming distance = 1
        for hd in range(d):
            diff_delE_nin=-2.0*x_nin[hd]*(x_nin[(hd+d-1)%d]+x_nin[(hd+1)%d])
            diff_E_nin=diff_delE_nin*theta_model
            gradK_nin+=diff_delE_nin*np.exp(0.5*diff_E_nin)
        gradK+=gradK_nin
    gradK*=(1.0/n_bach)
    theta_model=np.copy(theta_model) - lr * gradK
    theta_diff=abs(theta_model-J)
print("#theta_true=",J,"theta_estimated=",theta_model)

plt.hist(theta_vec,bins=10)
plt.title("Hist(J), Jack Knife")
plt.xlabel("J")
plt.ylabel("Hist(J)")
filename="hist_of_J4.png"
plt.savefig(filename)
