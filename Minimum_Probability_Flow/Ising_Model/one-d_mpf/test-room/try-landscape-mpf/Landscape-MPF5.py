import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
import csv 
np.random.seed(10)
t_interval = 40
d, N_sample =4,10 #124, 1000
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

def prob_x_goout(th,j,x=[]):#return: p(a -> b)
    pout=1.0/(1.0 + np.exp(2.0*th*x[j]*(x[(j+1)%d]+x[(j-1+d)%d]))) / d
    return  pout

def prob_x_into(th,j,x=[]):#return: p(a -> b)
    pin=1.0/(1.0 + np.exp( -2.0*th*x[j]*(x[(j+1)%d]+x[(j-1+d)%d]))) / d
    return pin 

#######    MAIN    ########
J_true=0.3
x = np.random.choice([-1,1],d)
for n in range(N_sample+N_remove):
    for t in range(t_interval):
        x = np.copy(gen_mcmc(J_true,x))
    if(n==N_remove):
        x_new=np.copy(x)
        X_sample = x_new
    elif(n>N_remove):
        x_new=np.copy(x)
        X_sample=np.vstack((X_sample,x_new))
print("X_sample=\n",X_sample)
#1================
dist_mat=d*np.ones((N_sample,N_sample))
dist_mat=2*np.copy(dist_mat)-2*(np.matrix(X_sample)*np.matrix(X_sample).T)
dist_mat/=4
idx=np.where(dist_mat!=1)
idx0=np.where(dist_mat==0)
dist_mat2=np.copy(dist_mat)
dist_mat2[idx]=0
dist_mat2[idx0]=-1
identical_xm=np.zeros(N_sample)
print("dis_mat2=\n",dist_mat2)
#1================
for u in range(N_sample):
    #2=====================
    idx2=np.where(dist_mat2.T[u]==1)
    idx02=np.where(dist_mat2.T[u]==-1)
    identical_xm[u]=len(idx02[0])
    check_list=np.zeros(d)       
    if(len(idx2[0]>0)):
        for i in idx2[0]:
            diff_sample_pair_i=X_sample[u]-X_sample[i]
            idx3=np.where(diff_sample_pair_i!=0)
            l2=idx3[0][0]
            check_list[l2]+=1
    #2=====================
    if(u==0):
        check_list_table=np.copy(check_list)
    elif(u>0):
        check_list_table=np.vstack((check_list_table,np.copy(check_list)))
print("identical_xm=\n",identical_xm)
print("check_list_table=\n",check_list_table)
bins=0.025
theta_slice=np.arange(-2.0,4.0,bins)
MPF_of_th=0
for th in theta_slice:
    #MCMC-mean(using CD-method)
    MPF_of_th_old=MPF_of_th
    MPF_of_th=0.0
    prob1=0
    for m in range(N_sample):
        xm=np.copy(X_sample[m])
        local_check_list=check_list_table[m]
        p_of_xm,q_of_xm=0.0,0.0
        for j in range(d):
            q_of_xm+=prob_x_goout(th,j,xm)# from xm
            p_of_xm+=local_check_list[j]*prob_x_into(th,j,xm) # to xm
        prob1_m=(p_of_xm+identical_xm[m]*(1-q_of_xm))/N_sample
        prob1+=prob1_m
        MPF_of_th+=np.log(prob1_m)/N_sample
    del_MPF = (MPF_of_th-MPF_of_th_old)/bins
    print(th,MPF_of_th,del_MPF,prob1)
