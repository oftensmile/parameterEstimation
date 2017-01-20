#! /usr/bin/env python
#-*-coding:utf-8-*-
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
from scipy.optimize import root 
np.random.seed(1)
n_estimation=4
#parameter ( MCMC )
d, N_sample =64,300#124, 1000
N_remove = 100
t_interval=30
lr,eps =0.5, 1.0e-100
t_gd_max=100 
def gen_mcmc(J=[],x=[]):
    for i in range(d):
        #Heat Bath
        diff_E=2.0*x[i]*(J[(i+d-1)%d]*x[(i+d-1)%d]+J[i]*x[(i+1)%d])#E_new-E_old
        r=1.0/(1+np.exp(diff_E)) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x

def calc_E(J=[],x=[]):
    e=0.0
    for i in range(d):
        e+=J[i]*x[i]*x[(i+1)%d]
    return e

def calc_C(X=[[]]):
    n_bach = len(X)
    corre_mean=np.zeros(d)
    for n in range(n_bach):
        xn=X[n]
        for i in range(d):
            #corre+=xn[i]*xn[(i+1)%d]/d
            corre_mean[i]+=xn[i]*xn[(i+1)%d]/n_bach
    return corre_mean

def Tk(J,k):
    l1=(2*np.cosh(J))**k
    l2=(2*np.sinh(J))**k
    return ( 0.5*(l1+l2) , 0.5*(l1-l2) )

#p(x_i=+1|x_1-i)
def gen_x_pofx(p_value):
    r=np.random.uniform()
    if(p_value>r):x_prop=1
    else:x_prop=-1
    return x_prop

def pofx_given_xprev(J,k,x_1,x_prev):
    ind_plus_prev=int(0.5*(1-x_prev)) #if same sign=>0
    ind_first_prev=int(0.5*(1-x_1*x_prev)) #if same sign=>0
    p=Tk(J,1)[ind_plus_prev] * Tk(J,d-k)[0] / Tk(J,d-k+1)[ind_first_prev]
    return p

def get_sample(j):
    X=np.zeros(d)
    #p(+)=p(-)=1/2
    X[0]=np.random.choice([-1,1])
    for k in range(1,d):
        p = pofx_given_xprev(j,k,X[0],X[k-1])
        X[k]=gen_x_pofx(p)
    return X

def grad_of_object(J,X_sample):
    diff_expect=np.zeros(d)
    for m in range(N_sample):
        x_m=np.copy(X_sample[m])
        for l in range(d):
            diff_E=2*x_m[l]*(J[l]*x_m[(l+1)%d]+J[(l-1+d)%d]*x_m[(l-1+d)%d])
            #next nerest
            diff_E_nn=2*x_m[(l+1)%d]*(J[(l+1)%d]*x_m[(l+2)%d]+J[l]*x_m[l])
            diff_expect[l]+=( - x_m[l]*x_m[(l+1)%d] )/N_sample *( np.exp(-0.5*diff_E)/d + np.exp(-0.5*diff_E_nn)/d )
    return diff_expect

if __name__ == '__main__':
    #fname="sample"+str(N_sample)+"nomcCD.dat"
    #f=open(fname,"w")
    #for nf in range(n_estimation):
    ##Generate sample-dist
    #J_data=np.random.rand(d) # =theta_sample
    #J_data=[0,1.0,1.0,0.5,0.0,1.0]
    J_data=np.random.uniform(0,1,d)
    correlation_data=0.0#np.zeros(d)
    #SAMPLING-Tmat                      1
    x=np.random.choice([-1,1],d)
    for n in range(N_sample+N_remove):
        for t in range(t_interval):
            x = np.copy(gen_mcmc(J_data,x))
        if(n==N_remove):
            x_new=np.copy(x)
            X_sample = np.copy(x)
            #correlation_data+=calc_C(x_new)/N_sample
        elif(n>N_remove):
            x_new=np.copy(x)
            X_sample=np.vstack((X_sample,np.copy(x)))
            #correlation_data+=calc_C(x_new)/N_sample
    J_model_init=np.random.uniform(0,2,d)
    J_root=root(grad_of_object,J_model_init,args=(X_sample),method="hybr")
    L=np.int(np.sqrt(d))
    J_model_mat=np.reshape(J_root.x,(L,L)) 
    J_data_mat=np.reshape(J_data,(L,L)) 
    
    plt.figure()
    plt.subplot(131)
    plt.imshow(J_model_mat, interpolation='nearest')
    plt.colorbar()
    plt.title("J_model")
    plt.subplot(132)
    plt.imshow(J_data_mat, interpolation='nearest')
    plt.colorbar()
    plt.title("J_data")
    plt.subplot(133)
    plt.imshow(J_model_mat-J_data_mat, interpolation='nearest')
    plt.colorbar()
    plt.title("J_model_mat-J_data_mat")
    plt.clim(-0.01,1)
    plt.show()
 
    
    
    """ 
    print("#solution=",J_root.x)
    print("#diff==",np.sum(np.abs(J_data-J_root.x)) )
    print("#check=",grad_of_object(J_root.x,X_sample))
    bins=np.arange(1,d+1)
    bar_width=0.2
    plt.bar(bins,J_data,color="blue",width=bar_width,label="$\it{J_{data}}$",align="center")
    plt.bar(bins+bar_width,J_root.x,color="red",width=bar_width,label="$\it{J_{moel}}$",align="center")
    #plt.bar(bins+2*bar_width,init_theta,color="green",width=bar_width,label="initial",align="center")
    plt.bar(bins+2*bar_width,J_model_init,color="gray",width=bar_width,label="$\it{J_{moel}};initial$",align="center")
    plt.legend(fontsize=18)
    plt.title("Minimum Probability Flow",fontsize=22)
    plt.xlabel("i=1,2,...,16",fontsize=18)
    plt.ylabel("J",fontsize=18)
    plt.show()    
 
   """ 
    
    
    #f.write(str(error)+"\n")
    #f.close()
