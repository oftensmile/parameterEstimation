#2016/05/19
##############
#   H = -sum(J_ij * xixj) - sum(h_i*xi), J in R^1
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
d, N_sample = 16,1000 #124, 1000
N_remove = 100
#parameter ( MPF+GD )
lr,eps =0.1, 1.0e-100
beta=1.0
def gen_mcmc(J=[],x=[] ):
    for i in range(d):
        #Heat Bath
        diff_E=beta*2.0*x[i]*(J[i]*x[(i+1)%d]+J[(i+d-1)%d]*x[(i+d-1)%d])
        r=1.0/(1+np.exp(diff_E)) 
        #r=np.exp(-diff_E) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x

#######    MAIN    ########
#Generate sample-dist
J_max,J_min=1,0.0
J_vec=np.random.uniform(J_min,J_max,d)
J_mat=np.zeros((d,d))
for i in range(d):
    J_mat[i][(i+1)%d]=J_vec[i]#*0.5
    J_mat[(i+1)%d][i]=J_vec[i]#*0.5
J_norm=np.sqrt(np.sum(np.sum(J_mat**2)))
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
    m[i]=np.sum(X_sample.T[i])/(n_bach)
m=np.matrix(m)

time_i=time.time()
XnXnt=np.zeros((d,d))
for n in range(n_bach):
    xn=np.matrix(X_sample[n])
    XnXnt=XnXnt+np.tensordot(xn,xn,axes=([0],[0]))/(n_bach)
mmt=np.tensordot(m,m,axes=([0],[0]))  

C=XnXnt-mmt
for l in range(d):
    C[l][l]=1.0
#SVD type
U,s,V = np.linalg.svd(C, full_matrices=True)
sinv=1.0/s
Cinv=np.dot(V.T,np.dot(np.diag(sinv),U.T))

J_hat=-Cinv
#this is a tryal 
for i in range(d):
    J_hat[i][i]=0
J_hat_norm=np.sqrt(np.sum(np.sum(J_hat**2)))
#Estimated matrix is normalized
J_hat=np.copy(J_hat)*(J_norm/J_hat_norm)
error_mat=J_hat-J_mat
#error=np.sum(np.sum(np.abs(J_hat-J_mat)))/(d*d)
error=np.sqrt(np.sum(np.sum((J_hat-J_mat)**2)))/J_norm

print("error = ",error)
time_f=time.time()
dtime=time_f-time_i
print("calc time =",dtime)
#visualize
plt.figure()
plt.subplot(131)
plt.imshow(J_mat,interpolation='nearest')
plt.colorbar()
plt.title("True J")
plt.subplot(132)
plt.imshow(J_hat,interpolation='nearest')
plt.colorbar()
plt.title("Estimated J")
plt.subplot(133)
plt.imshow(error_mat,interpolation='nearest')
plt.colorbar()
plt.title("J_hat-J_true")
plt.show()
