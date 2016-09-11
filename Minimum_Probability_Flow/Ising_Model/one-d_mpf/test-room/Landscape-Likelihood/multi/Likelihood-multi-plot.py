import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
import csv 
#parameter ( MCMC )
t_interval = 40
d, N_sample =16,3000 #124, 1000
num_mcmc_sample=500
N_remove = 100
lr,eps =0.1, 1.0e-100
t_gd_max=200 
def gen_mcmc(J,x=[] ):
    for i in range(d):
        #Heat Bath
        diff_E=2.0*x[i]*J*(x[(d+1+i)%d]+x[(i+d-1)%d])
        r=1.0/(1+np.exp(diff_E)) 
        #r=np.exp(-diff_E) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x

def calc_E(x_tot=[[]],theta=[[]]):
    len_x=len(x_tot)
    E=np.zeros(len_x)
    for n in range(len_x):
        x_n=np.copy(x_tot[n])
        E[n]=np.matrix(x_n)*np.matrix(theta)*np.matrix(x_n).T
    return E

#######    MAIN    ########
##Generate sample-dist
J_true=0.5
total_set=10
for my_set in range(total_set):
    f=open("liklihood-"+str(my_set)+".dat", "w")
    np.random.seed(my_set)
    x = np.random.choice([-1,1],d)
    correlation_data=np.zeros(d)
    ##SAMPLING
    for n in range(N_sample+N_remove):
        for t in range(t_interval):
            x = np.copy(gen_mcmc(J_true,x))
        if(n==N_remove):
            x_new=np.copy(x)
            for i in range(d):
                correlation_data[i]=x_new[i]*x_new[(i+1)%d]/N_sample
            X_sample = x_new
        elif(n>N_remove):
            x_new=np.copy(x)
            for i in range(d):
                correlation_data[i]+=x_new[i]*x_new[(i+1)%d]/N_sample
            X_sample=np.vstack((X_sample,x_new))
     
    ######### L(theta)=sum( theta*sum(xixj) - log((2cosh(theta))**d+(2cosh(theta))**d) ) #########
    theta_model=2.0
    bins=0.001
    theta_slice=np.arange(.4,.6,bins)
    sum_correlation_data=np.sum(correlation_data)
    sum_prob=0
    record=np.zeros(len(theta_slice))
    k=0
    for th in theta_slice:
        #MCMC-mean(using CD-method)
        l_of_theta=(th*sum_correlation_data-np.log( (2*np.cosh(th))**d + (2*np.sinh(th))**d) )
        record[k]=l_of_theta
        k+=1
        #sum_prob+=np.exp(l_of_theta)*bins
        #print(th,l_of_theta,np.exp(l_of_theta),sum_prob)
    max_l_of_theta=np.abs(np.max(record))
    for th in theta_slice:
        #print(th,l_of_theta/max_l_of_theta)
        l_of_theta=(th*sum_correlation_data-np.log( (2*np.cosh(th))**d + (2*np.sinh(th))**d) )
        f.write(str(th)+"  "+str(l_of_theta/max_l_of_theta)+"\n")
    f.close()
