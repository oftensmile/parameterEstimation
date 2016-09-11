#2016/08/05
##############
#   H = -J*sum(xixj), J in R^1
##############
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
from scipy import optimize
#import csv 
np.random.seed(10)
#parameter ( MCMC )
#t_burn_emp, t_burn_model = 1100, 10#10000, 100
t_interval = 40
#parameter ( System )
#d, N_sample = 16,300 #124, 1000
d, N_sample =16,300 #124, 1000
#N_remove = 100
N_remove = 100
#parameter ( MPF+GD )
lr,eps =0.1, 1.0e-100
t_gd_max=300 
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


def cost_func(theta_model=[]):
    K=.0
    for nin in range(N_sample):
        x_nin=np.copy(X_sample[nin])
        #gradK_nin=np.zeros(d)
        #idx2=np.where(dist_mat2.T[nin]==1)
        check_list=np.zeros(d)
        """ 
        #Extracting only balanced flow.
        normalize=d
        if(len(idx2[0]>0)):
            for i in idx2[0]:
                diff_sample_pair_i=X_sample[nin]-X_sample[i]
                idx3=np.where(diff_sample_pair_i!=0)
                l2=idx3[0][0]
                if(check_list[l2]==0):
                    normalize-=1
                    check_list[l2]+=1
        """
        K_nin=.0
        for l in range(d):
            if(check_list[l]==0):
                xl_xl_plu_1=x_nin[l]*x_nin[(l+1)%d]
                xl_min_1_xl=x_nin[(l+d-1)%d]*x_nin[l]
                K_nin+=np.exp(xl_xl_plu_1*theta_model[l]+xl_min_1_xl*theta_model[(l-1+d)%d])
        K+=K_nin/N_sample
    return K



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
theta_model=np.random.uniform(.0,4.0,d)
print("diff_J-true_J-init=",np.sqrt(np.sum((theta_model-J_vec)**2)))
theta_ans=optimize.fmin_cg(cost_func,theta_model)
print("calcuration of K = ", cost_func(J_vec))
print("diff_J-true_J-ans=",np.sqrt(np.sum((theta_ans-J_vec)**2)))
