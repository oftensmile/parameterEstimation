#2016/08/05
##############
#   H = -J*sum(xixj), J in R^1
##############
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
import csv 
np.random.seed(11)
#parameter ( MCMC )
t_interval = 40
#d, N_sample =16,200 #124, 1000
d, N_sample =5,10 #124, 1000
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

def prob_to_x(j,x=[],th=[]):#return: p(a -> b)
    return 1.0/(1.0 + np.exp(-2.0*th*x[j]*(x[(j+1)%d]+x[(j-1+d)%d]))) / d 

def prob_from_x(j,x=[],th=[]):#return: p(a -> b)
    return 1.0/(1.0 + np.exp(2.0*th*x[j]*(x[(j+1)%d]+x[(j-1+d)%d]))) / d
#######    MAIN    ########
J_true=0.3
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
dist_mat2=np.copy(dist_mat)
dist_mat2[idx]=0
dist_mat2[idx0]=-1
identical_xm=np.zeros(N_sample)
identical_wiout=0
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
            #if(check_list[l2]==0):
               # check_list[l2]+=1
            check_list[l2]+=1
    #2=====================
    if(u==0):
        check_list_table=np.copy(check_list)
        without_doublecount=np.copy(X_sample[u])
        identical_wiout=len(idx02[0])
    elif(u>0):
        check_list_table=np.vstack((check_list_table,np.copy(check_list)))
        if(len(np.where(idx02[0]<u)[0])==0):#This state doesn't happen before u.
            without_doublecount=np.vstack((without_doublecount,np.copy(X_sample[u])))
            identical_wiout=np.append(identical_wiout,len(idx02[0]))
theta_model=2.0
bins=0.025
theta_slice=np.arange(-2.0,2.0,bins)
sum_correlation_data=np.sum(correlation_data)
MPF_of_th=0
count=0
size_identical_wiout=len(identical_wiout)
for th in theta_slice:
    count+=1
    #MCMC-mean(using CD-method)
    MPF_of_th_old=MPF_of_th
    MPF_of_th=0.0
    prob1=0
    for m in range(size_identical_wiout):
        xm=np.copy(without_doublecount[m])
        p_xm_into,p_xm_goout=0.0,0.0
        for j in range(d):
            p_xm_goout+=prob_from_x(j,xm,th)
        for 







    for m in range(N_sample):
        #p_of_xm:Hamming 1  => data to another data
        #q_of_xm:Hamming 0  => data to owne data
        xm=np.copy(X_sample[m])
        local_check_list=check_list_table[m]
        p_of_xm,q_of_xm=0.0,0.0
        for j in range(d):
            q_of_xm+=prob_from_x(j,xm,th)
            if(local_check_list[j]>0):
                p_of_xm+=local_check_list[j] * prob_to_x(j,xm,th)# to xm
        prob1_m=(p_of_xm+(1-q_of_xm)*identical_xm[m] )/N_sample
        
        prob1+=prob1_m
        MPF_of_th+=np.log(prob1_m)/N_sample
    del_MPF = (MPF_of_th-MPF_of_th_old)/bins
    print(th,MPF_of_th,del_MPF,prob1)
