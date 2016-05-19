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
t_interval = 3
#parameter ( System )
d, N_sample = 4, 10#124, 1000
#parameter ( MPF+GD )
eps = 0.1

theta=[[1 if i==(j+1+d)%d or i==(j-1+d)%d else 0 for i in range(d)] for j in range(d)]
theta=np.array(theta)
theta_model = np.arange(d*d)
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

#Create different 'd' states, which different 1bit from given x.

grad_K=np.zeros((d,d))
for i in range(N_sample):

    #a=2 # temporal index
    A=np.ones((d,d))
    np.fill_diagonal(A,-1)
    #x_test=[1,-1,1,-1]
    xi=np.copy(X_sample[i,:])
    #x_outer=np.outer(x_test,x_test)
    x_outer=np.outer(xi,xi)
    temp=x_outer*theta_model
    temp2=np.dot(np.dot(A,temp),A)
    temp2=0.5*temp2
    temp3=np.exp(temp2)
    for j in range(d):
        
        partial_x=np.zeros((d,d))
        partial_x[j,:]=np.copy(x_outer[j,:])
        partial_x[:,j]=np.copy(x_outer[:,j])
        partial_x[j,j]=0
        grad_K=grad_K+partial_x*temp3[j,j]
    grad_K=garad_K*exp(-0.5Ei)


#Generate sample
x = np.ones(d)
#BURN-IN 
for t_burn in range(t_burn_emp):
    x = np.copy(gen_mcmc(x))
#SAMPLING
for n in range(N_sample):
    for t in range(t_interval):
        x = np.copy(gen_mcmc(x))
    if(n==0):X_sample = np.copy(gen_mcmc(x))
    elif(n>0):X_sample=np.vstack((X_sample,x))
########    MAIN    ########
Ene_of_x = np.zeros(N_sample)
print("part of X_sample =",X_sample[1,:])
for i in range(N_sample):
    xi=np.copy(X_sample[i,:])
    Ene_of_x[i]=np.dot(xi,np.dot(xi,theta))













#output equilibrium state data
"""
FILE = "sample.csv"
f=open(FILE,"w")
c=csv.writer(f)
c.writerows(X_sample)
f.close()
"""

