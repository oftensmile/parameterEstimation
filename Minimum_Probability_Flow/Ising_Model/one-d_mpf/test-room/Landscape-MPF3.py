#2016/08/05
##############
#   H = -J*sum(xixj), J in R^1
##############
import numpy as np
import time
from scipy import linalg
import matplotlib.pyplot as plt
import csv 
np.random.seed(10)
#parameter ( MCMC )
t_interval = 40
#d, N_sample =16,200 #124, 1000
d, N_sample =16,30 #124, 1000
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
J_true=0.5
x = np.random.choice([-1,1],d)
correlation_data=np.zeros(d)
##SAMPLING
sum_correlation_data_vec=np.zeros(N_sample)
for n in range(N_sample+N_remove):
    for t in range(t_interval):
        x = np.copy(gen_mcmc(J_true,x))
    if(n==N_remove):
        x_new=np.copy(x)
        for i in range(d):
            correlation_data[i]=x_new[i]*x_new[(i+1)%d]/N_sample
            sum_correlation_data_vec[n-N_remove]+=x_new[i]*x_new[(i+1)%d]
        X_sample = x_new
    elif(n>N_remove):
        x_new=np.copy(x)
        for i in range(d):
            correlation_data[i]+=x_new[i]*x_new[(i+1)%d]/N_sample
            sum_correlation_data_vec[n-N_remove]+=x_new[i]*x_new[(i+1)%d]
        X_sample=np.vstack((X_sample,x_new))

#1================
dist_mat=d*np.ones((N_sample,N_sample))
dist_mat=2*np.copy(dist_mat)-2*(np.matrix(X_sample)*np.matrix(X_sample).T)
dist_mat/=4
idx=np.where(dist_mat!=1)
idx0=np.where(dist_mat==0)
for j in range(N_sample):
    print(j,"-th 0 entry is ", idx0[j])
dist_mat2=np.copy(dist_mat)
dist_mat2[idx]=0 
#1================
for u in range(N_sample):
    #2=====================
    idx2=np.where(dist_mat2.T[u]==1)
    check_list=np.zeros(d)       
    if(len(idx2[0]>0)):
        for i in idx2[0]:
            diff_sample_pair_i=X_sample[u]-X_sample[i]
            idx3=np.where(diff_sample_pair_i!=0)
            l2=idx3[0][0]
            #if(check_list[l2]==0):
               # check_list[l2]+=1
            check_list[l2]+=1
    #2=====================
    if(u==0):
        check_list_table=np.copy(check_list)
    elif(u>0):
        check_list_table=np.vstack((check_list_table,np.copy(check_list)))

theta_model=2.0
bins=0.025
theta_slice=np.arange(-2.0,2.0,bins)
sum_correlation_data=np.sum(correlation_data)
MPF_of_th=0
for th in theta_slice:
    #MCMC-mean(using CD-method)
    MPF_of_th_old=MPF_of_th
    MPF_of_th=0.0
    for m in range(N_sample):
        #CD_of_th_m=th*sum_correlation_data_vec[m]-np.log( (2*np.cosh(th))**d + (2*np.sinh(th))**d)
        #p_of_xm:Hamming 1  => data to another data
        #q_of_xm:Hamming 0  => data to owne data
        p_of_xm,q_of_xm=0.0,0.0
        xm=np.copy(X_sample[m])
        local_check_list=check_list_table[m]
        for j in range(d):
            q_of_xm+=1.0/(1.0 + np.exp(2.0*th*xm[j]*(xm[(j+1)%d]+xm[(j-1+d)%d]))) / d
            if(local_check_list[j]!=0):
                p_of_xm+=local_check_list[j]/(1.0 + np.exp(-2.0*th*xm[j]*(xm[(j+1)%d]+xm[(j-1+d)%d]))) / d
        MPF_of_th+=-np.log(p_of_xm+1.0-q_of_xm)/N_sample
        #   Using sugestion.
        #MPF_of_th+=-np.log((p_of_xm+ 1.0-q_of_xm))/N_sample
    delta_MPF = (MPF_of_th-MPF_of_th_old)/bins
    print(th,MPF_of_th,delta_MPF,np.exp(MPF_of_th))
