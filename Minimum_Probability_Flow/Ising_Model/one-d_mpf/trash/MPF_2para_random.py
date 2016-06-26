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
d, N_sample = 64,1000 #124, 1000
#parameter ( MPF+GD )
lr,eps =0.01, 1.0e-100
n_mfa = 100 #Number of the sample for Mean Field Aproximation.
t_gd_max=100 
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
J1,J2=1.5,1.2 # =theta_sample
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

theta_model1,theta_model2=3.0, 2.0  #Initial Guess
print("#gd-step, abs-grad_likelihood, theta-error")
theta_slice=200
theta_min,theta_max=-2,2
dtheta=(theta_max-theta_min)*(1.0/theta_slice)
for t in range(theta_slice):
    theta_model1+=theta_min+t*dtheta
    theta_model2+=theta_min*t*dtheta
    KofTheta1,KofTheta2=0.0,0.0
    n_bach=len(X_sample)
    for nin in range(n_bach):
        x_nin=X_sample[nin]
        KofTheta1_nin,KofTheta2_nin=0.0,0.0
        gradK1_nin,gradK2_nin=0.0,0.0
        #hamming distance = 1
        for hd in range(d):
            diff_delE1_nin=-2.0*x_nin[hd]*(x_nin[(hd+d-1)%d]+x_nin[(hd+1)%d])
            diff_delE2_nin=-2.0*x_nin[hd]*(x_nin[(hd+d-2)%d]+x_nin[(hd+2)%d])
            diff_E1_nin=diff_delE1_nin*theta_model1/d
            diff_E2_nin=diff_delE2_nin*theta_model2/d
            diff_E_nin=diff_E1_nin+diff_E2_nin
            #gradK1_nin+=diff_delE1_nin*np.exp(0.5*diff_E_nin)
            #gradK2_nin+=diff_delE2_nin*np.exp(0.5*diff_E_nin)
            KofTheta1_nin+=np.exp(0.5*diff_E1_nin)
            KofTheta2_nin+=np.exp(0.5*diff_E2_nin)
        #gradK1+=gradK1_nin/n_bach
        #gradK2+=gradK2_nin/n_bach
        KofTheta1+=KofTheta1_nin/n_bach
        KofTheta2+=KofTheta2_nin/n_bach
    #gradK1*=(1.0/n_bach)
    #gradK2*=(1.0/n_bach)
    #print(gradK1,gradK2)
    print(theta_model1,KofTheta1,KofTheta2)
    #theta_model1=np.copy(theta_model1) - lr * gradK1
    #theta_model2=np.copy(theta_model2) - lr * gradK2
    #theta_diff1=abs(theta_model1-J1)
    #theta_diff2=abs(theta_model2-J2)
    #print(t_gd,np.abs(gradK1),np.abs(gradK2),theta_diff1,theta_diff2)
print("#theta1,theta2 (true)=",J1,J2,"theta1,theta2 _estimated=",theta_model1,theta_model2)

