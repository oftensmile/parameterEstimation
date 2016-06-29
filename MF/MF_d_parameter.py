#2016/05/19
##############
#   H = -J*sum(xixj), J in R^1
##############
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
import csv 
from numpy.linalg import inv
np.random.seed(0)
#parameter ( MCMC )
#t_burn_emp, t_burn_model = 1100, 10#10000, 100
t_interval = 40
#parameter ( System )
d, N_sample = 16,300 #124, 1000
N_remove = 100
#parameter ( MPF+GD )
lr,eps =0.1, 1.0e-100
t_gd_max=140 
def gen_mcmc(J=[],x=[] ):
    for i in range(d):
        #Heat Bath
        diff_E=2.0*x[i]*(J[i]*x[(d+1)%d]+J[(i+d-1)%d]*x[(i+d-1)%d])
        r=1.0/(1+np.exp(diff_E)) 
        #r=np.exp(-diff_E) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x

#######    MAIN    ########
#Generate sample-dist
J_max,J_min=5,0.0
J_vec=np.random.uniform(J_min,J_max,d)
J_mat=np.zeros((d,d))
for i in range(d):
    J_mat[i][(i+1)%d]=J_vec[i]*0.5
    J_mat[(i+1)%d][i]=J_vec[i]*0.5

x = np.random.uniform(-1,1,d)
x = np.array(np.sign(x))
#SAMPLING
for n in range(N_sample):
    for t in range(t_interval):
        x = np.copy(gen_mcmc(J_vec,x))
    if(n==N_remove):X_sample = np.copy(x)
    elif(n>N_remove):X_sample=np.vstack((X_sample,np.copy(x)))
#MF
m=np.zeros(d)
n_bach=len(X_sample)
for i in range(d):
    m[i]=np.sum(X_sample.T[i])/n_bach
m=np.matrix(m)

time_i=time.time()
XnXnt=np.zeros((d,d))
for n in range(n_bach):
    xn=np.matrix(X_sample[n])
    XnXnt=XnXnt+np.tensordot(xn,xn,axes=([0],[0]))/n_bach
mmt=np.tensordot(m,m,axes=([0],[0]))  
C=XnXnt-mmt
for l in range(d):C[l][l]=0.0
#Cinv=inv(C)
#SVD type
U,s,V = np.linalg.svd(C, full_matrices=True)
sinv=1.0/s
Cinv=np.dot(V.T,np.dot(np.diag(sinv),U.T))

J_hat=-Cinv
error_mat=J_hat-J_mat
error=np.sum(np.sum(np.abs(J_hat-J_mat)))/(d*d)

print("error = ",error)
time_f=time.time()
dtime=time_f-time_i
print("calc time =",dtime)
#visualize
plt.figure()
plt.subplot(131)
plt.imshow(J_mat)
plt.colorbar()
plt.title("True J")
plt.subplot(132)
plt.imshow(J_hat)
plt.colorbar()
plt.title("Estimated J")
plt.subplot(133)
plt.imshow(error_mat)
plt.colorbar()
plt.title("J_hat-J_true")
plt.show()
