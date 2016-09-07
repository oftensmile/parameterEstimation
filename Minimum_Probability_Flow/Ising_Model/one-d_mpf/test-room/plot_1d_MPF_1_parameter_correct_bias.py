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
#parameter ( System )
d, N_sample =16,7200 #124, 1000
N_remove = 100
#parameter ( MPF+GD )
lr,eps =0.1, 1.0e-100
t_gd_max=200 
def gen_mcmc(J,x=[] ):
    for i in range(d):
        #Heat Bath
        diff_E=2.0*J*x[i]*(x[(d+1+i)%d]+x[(i+d-1)%d])
        r=1.0/(1+np.exp(diff_E)) 
        #r=np.exp(-diff_E) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x

#######    MAIN    ########
##Generate sample-dist
#J_vec=np.random.uniform(J_min,J_max,d)
J_tru=0.3
x = np.random.uniform(-1,1,d)
x = np.array(np.sign(x))
##SAMPLING
for n in range(N_sample+N_remove):
    for t in range(t_interval):
        x = np.copy(gen_mcmc(J_tru,x))
    if(n==N_remove):X_sample = np.copy(x)
    elif(n>N_remove):X_sample=np.vstack((X_sample,np.copy(x)))

##Find Hamming Distance 1 whith in the data.
dist_mat=d*np.ones((N_sample,N_sample))
dist_mat=2*np.copy(dist_mat)-2*(np.matrix(X_sample)*np.matrix(X_sample).T)
dist_mat/=4
idx=np.where(dist_mat!=1)
dist_mat2=np.copy(dist_mat)
dist_mat2[idx]=0    #Dist of none zero elemens are 1 hamming distance.

#In this case I applied 
theta_model=1.5   #Initial guess
n_bach=len(X_sample)
theta_slice=np.arange(-1,1,0.05)
#for t_gd in range(t_gd_max):

for ts in theta_slice:
    K,gradK=0.0,0.0
    for nin in range(n_bach):
        x_nin=np.copy(X_sample[nin])
        gradK_nin=0
        #idx2=np.where(dist_mat2[nin]==1)
        idx2=np.where(dist_mat2.T[nin]==1)
        check_list=np.zeros(d)
        #"""
        normalize=d
        if(len(idx2[0]>0)):
            for i in idx2[0]:
                diff_sample_pair_i=X_sample[nin]-X_sample[i]
                idx3=np.where(diff_sample_pair_i!=0)
                l2=idx3[0][0]
                if(check_list[l2]==0):
                    normalize-=1
                    check_list[l2]+=1
        #"""
        for l in range(d):
            if(check_list[l]==0):
                gradE_nin=-x_nin[l]*(x_nin[(l+1)%d]+ x_nin[(l-1+d)%d])
                gradK_nin=gradE_nin*np.exp(ts*gradE_nin)/d
                K+=np.exp(ts*gradE_nin)/(n_bach*d)
                gradK+=gradK_nin/n_bach
                   
    #theta_model-=lr*gradK
    #sum_of_gradK=gradK
    #error_func=np.abs(ts-J_tru)
    print(ts,gradK,K)
#Plot
"""
bins=np.arange(1,d+1)
bar_width=0.2
plt.bar(bins,J_vec,color="blue",width=bar_width,label="true",align="center")
plt.bar(bins+bar_width,theta_model,color="red",width=bar_width,label="estimated",align="center")
plt.bar(bins+2*bar_width,init_theta,color="green",width=bar_width,label="initial",align="center")
plt.bar(bins+3*bar_width,gradK*10,color="gray",width=bar_width,label="gradK",align="center")
plt.legend()
plt.title("With bias correction")
#filename="test_output_fixed6.png"
#plt.savefig(filename)
plt.show()
"""
