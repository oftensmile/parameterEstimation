#2016/05/19
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
import csv
np.random.seed(0)
#parameter ( Model )
J = 0.1
#parameter ( MCMC )
t_burn_emp, t_burn_model = 1000, 10#10000, 100
t_interval = 10
#parameter ( System )
d, N_sample = 3, 100#124, 1000
#parameter ( MPF+GD )
eps = 0.01
theta=[[1 if i==(j+1+d)%d or i==(j-1+d)%d else 0 for i in range(d)] for j in range(d)]
theta=np.array(theta)
theta_model = np.reshape(np.ones(d*d),(d,d))
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

#Initial
x = np.ones(d)
#BURN-IN 
for t_burn in range(t_burn_emp):
    #x = np.copy(gen_mcmc(x))
    x = gen_mcmc(x)
#SAMPLING
for n in range(N_sample):
    for t in range(t_interval):
        #x = np.copy(gen_mcmc(x))
        x = gen_mcmc(x)
    #if(n==0):X_sample = np.copy(gen_mcmc(x))
    if(n==0):X_sample = gen_mcmc(x)
    elif(n>0):X_sample=np.vstack((X_sample,x))
########    MAIN    ########
for t in range(10):
    #print("part of X_sample=",X_sample[1,:])
    grad_K=np.zeros((d,d))
    for i in range(N_sample):
        xi=np.copy(X_sample[i,:])
        Ei=np.dot(xi,np.dot(theta_model,xi))/d
        A=np.ones((d,d))
        np.fill_diagonal(A,-1)
        temp1=xi*A
        for j in range(d):
            xj=np.copy(temp1[:,j])
            grad_K=grad_K+0.5*(np.outer(xi,xi)-np.outer(xj,xj))*np.exp(0.5*np.dot(xj,np.dot(theta_model,xj)))
        grad_K=grad_K*np.exp(-0.5*np.dot(xi,np.dot(theta_model,xi)))
    #print(grad_K)
    grad_K=(1.0/N_sample)*grad_K
# it need update of theta
    #print("just chack of the one update of theta \n grad_K=\n",grad_K) 
    theta_model=theta_model+eps*grad_K
    #print("theta_model = \n", theta_model)
    print("grad_K= \n", grad_K)
print("theta_model = \n", theta_model)
print("theta = \n " , theta)
#output equilibrium state data
"""
FILE = "sample.csv"
f=open(FILE,"w")
c=csv.writer(f)
c.writerows(X_sample)
f.close()
"""

