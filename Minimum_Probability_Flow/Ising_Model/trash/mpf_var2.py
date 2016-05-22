#2016/05/19
import numpy as np
import time 
from scipy import linalg
import numpy.matlib
import matplotlib.pyplot as plt
import csv
np.random.seed(0)
#parameter ( Model )
J = 0.1
#parameter ( MCMC )
t_burn_emp, t_burn_model = 1000, 10#10000, 100
t_interval = 10
#parameter ( System )
d, N_sample = 3, 4#124, 1000
#parameter ( MPF+GD )
eps = 0.01
theta=[[1 if i==(j+1+d)%d or i==(j-1+d)%d else 0 for i in range(d)] for j in range(d)]
theta=np.array(theta)
theta_model = np.ones(d*d)
theta_model = np.reshape(theta_model,(d,d))#np.ones((d,d))
def gen_mcmc(x=[]):
    for i in range(d):
        valu=J*(np.dot(theta[i,:],x)-x[i]*theta[i][i])
        r=np.exp(-valu)/(np.exp(-valu)+np.exp(valu))
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=1
        else:
            x[i]=-1
        return x

#Generate sample
x = np.ones(d)
#BURN-IN 
for t_burn in range(t_burn_emp):
    x = gen_mcmc(x)
#SAMPLING
for n in range(N_sample):
    for t in range(t_interval):
        x = gen_mcmc(x)
    if(n==0):X_sample = gen_mcmc(x)
    elif(n>0):X_sample=np.vstack((X_sample,x))
########    MAIN    ########
Id=np.eye(d)
for t in range(100):
    grad_K=np.zeros((d,d))
    for i in range(N_sample):
        xi=np.copy(X_sample[i,:])
        x_outer=np.outer(xi,xi)
        xi_theta_1=np.dot(np.ones(d), np.dot(theta_model,xi))
        a = np.reshape(xi,(d,1))*xi_theta_1 - np.reshape(np.array(np.diag(theta_model)),(d,1))
        exp_a = np.exp ( a*(1.0/d))
        A = np.matlib.repmat(exp_a,1,d) + np.matlib.repmat(exp_a.T,d,1)
        array_exp_a = np.array(exp_a)
        #temp_mat is beter than temp_mat2
        #temp_mat =np.copy(np.diag(array_exp_a))
        temp_mat2=np.zeros((d,d))
        for l in range(d):
            temp_mat2[l][l]=array_exp_a[l]
        #B =np.multiply(A , x_outer) - np.diag(array_exp_a)
        B =np.multiply(A , x_outer) - temp_mat2
        grad_K = grad_K + B
    grad_K=(1.0/N_sample)*grad_K
    theta_model=theta_model-eps*grad_K
print("theta_model = \n", theta_model)
print("theta = \n " , theta)
#output equilibrium state data
"""
FILE = "sample.csv"
f=open(FIsLE,"w")
c=csv.writer(f)
c.writerows(X_sample)
f.close()
"""
