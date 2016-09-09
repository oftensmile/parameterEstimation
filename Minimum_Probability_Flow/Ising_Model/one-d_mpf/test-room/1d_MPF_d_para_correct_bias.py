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
d, N_sample =16,300 #124, 1000
N_remove = 100
#parameter ( MPF+GD )
lr,eps =0.1, 1.0e-100
t_gd_max=200 
def gen_mcmc(J=[],x=[] ):
    for i in range(d):
        #Heat Bath
        diff_E=2.0*x[i]*(J[i]*x[(d+1+i)%d]+J[(i+d-1)%d]*x[(i+d-1)%d])
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
J_max,J_min=1.0,0.0
J_vec=np.random.uniform(J_min,J_max,d)

x = np.random.uniform(-1,1,d)
x = np.array(np.sign(x))
##SAMPLING
for n in range(N_sample+N_remove):
    for t in range(t_interval):
        x = np.copy(gen_mcmc(J_vec,x))
    if(n==N_remove):X_sample = np.copy(x)
    elif(n>N_remove):X_sample=np.vstack((X_sample,np.copy(x)))
###MPF

##Find Hamming Distance 1 whith in the data.
dist_mat=d*np.ones((N_sample,N_sample))
dist_mat=2*np.copy(dist_mat)-2*(np.matrix(X_sample)*np.matrix(X_sample).T)
dist_mat/=4
idx=np.where(dist_mat!=1)
dist_mat2=np.copy(dist_mat)
dist_mat2[idx]=0    #Dist of none zero elemens are 1 hamming distance.
#In this case I applied 
theta_model=np.random.uniform(0,4,d)    #Initial guess
init_theta=np.copy(theta_model)
n_bach=len(X_sample)
for t_gd in range(t_gd_max):
    gradK=np.zeros(d)
    for nin in range(n_bach):
        x_nin=np.copy(X_sample[nin])
        #print("X_sample[nin]=\n",x_nin)
        gradK_nin=np.zeros(d)
        #idx2=np.where(dist_mat2[nin]==1)
        idx2=np.where(dist_mat2.T[nin]==1)
        #To eliminate transiton of data to data.
        #idx2 includes index of sample.
        check_list=np.zeros(d)
        #""" 
        #Extracting only balanced flow.
        normalize=d
        if(len(idx2[0]>0)):
            for i in idx2[0]:
                diff_sample_pair_i=X_sample[nin]-X_sample[i]
                idx3=np.where(diff_sample_pair_i!=0)
                l2=idx3[0][0]
                #print("x_smaple[",i,"]=\n",X_sample[i]," l2=",l2)
                if(check_list[l2]==0):
                    normalize-=1
                    #gradK[l2]-=gradK_nin[l2]/n_bach
                    check_list[l2]+=1
        for l in range(d):
            if(check_list[l]==0):
            #if(check_list[l]==1):
                xl_xl_plu_1=x_nin[l]*x_nin[(l+1)%d]
                xl_min_1_xl=x_nin[(l+d-1)%d]*x_nin[l]
                xl_plu_1_xl_pul_2=x_nin[(l+1)%d]*x_nin[(l+2)%d]
                gradK_nin[l]= - xl_xl_plu_1*np.exp( -xl_xl_plu_1*theta_model[l] ) /d *2
                gradK_nin[l]*= ( np.exp(-xl_min_1_xl*theta_model[(l+d-1)%d])+np.exp(-xl_plu_1_xl_pul_2*theta_model[(l+1)%d]) )
                gradK[l]+=gradK_nin[l]/n_bach
                   
    theta_model=theta_model-lr*gradK
    sum_of_gradK=np.sum(gradK)
    error_func=np.sum(np.abs(theta_model-J_vec))/d
    print(t_gd,sum_of_gradK,error_func)

