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
d, N_sample = 16,1000 #124, 1000
#parameter ( MPF+GD )
lr,eps =0.01, 1.0e-100
n_mfa = 100 #Number of the sample for Mean Field Aproximation.
t_gd_max=200 
def gen_mcmc(J1,J2,x=[] ):
    for i in range(d):
        #Heat Bath
        #diff_E=2.0*J*x[i]*(x[(i+d-1)%d]+x[(i+1)%d])#E_new-E_old
        diff_E=2.0*x[i]*( J1*(x[(i+d-1)%d]+x[(i+1)%d]) + J2* (x[(i+d-2)%d]+x[(i+2)%d]) ) 
        r=1.0/(1+np.exp(diff_E)) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x

########    MAIN    ########
#Generate sample-dist
J1,J2=0.0,10.0 # =theta_sample
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

#theta_model=4.0   #Initial Guess
theta_slice=200
theta_min,theta_max=-2,2
dtheta=(theta_max-theta_min)*(1.0/theta_slice)
print("#theta, KofTheta")
for t in range(theta_slice):
    #calc gradK of theta
    theta_model=theta_min+t*dtheta
    #KofTheta1,KofTheta2=0.0,0.0
    KofTheta=0.0
    n_bach=len(X_sample)
    for nin in range(n_bach):
        x_nin=X_sample[nin]
        #KofTheta_nin1,KofTheta_nin2=0.0,0.0
        KofTheta_nin=0.0
        #hamming distance = 1
        for hd in range(d):
            diff_delE_nin1=-2.0*x_nin[hd]*(x_nin[(hd+d-1)%d]+x_nin[(hd+1)%d])
            diff_delE_nin2=-2.0*x_nin[hd]*(x_nin[(hd+d-2)%d]+x_nin[(hd+2)%d])
            diff_E_nin1=diff_delE_nin1*theta_model
            diff_E_nin2=diff_delE_nin2*theta_model
            diff_delE_nin1=0.0
            KofTheta_nin+=np.exp(0.5*(diff_E_nin1+diff_E_nin2))
        KofTheta+=KofTheta_nin
        #KofTheta2+=KofTheta_nin2
    KofTheta=KofTheta*(1.0/n_bach)
    print(theta_model,KofTheta)
print("#theta_true=",J1,J2,"theta_estimated=",theta_model)
