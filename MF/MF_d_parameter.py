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
d, N_sample = 8,800 #124, 1000
N_remove = 200
#parameter ( MPF+GD )

J=np.random.choice([-1,1],(d,d))
for i in range(d):
    for j in range(i,d):
        if(j==i):
            J[i][i]=0.0
        else:
            J[j][i]=J[i][j]

beta=1.0
"""
#def gen_mcmc(J=[],x=[]):
#    for i in range(d):
#        #Heat Bath
#        diff_E=2.0*x[i]*(J[i]*x[(i+1)%d]+J[(i+d-1)%d]*x[(i+d-1)%d])
#        r=1.0/(1+np.exp(diff_E)) 
#        #r=np.exp(-diff_E) 
#        R=np.random.uniform(0,1)
#        if(R<=r):
#            x[i]=x[i]*(-1)
#    return x
"""

def gen_mcmc(x=[]):
    #Heat Bath
    for i in range(d):
        diff_E=0.0
        diff_E=beta*x[i]*(np.dot(J[i],x)+np.dot(J.T[i],x))
        r=1.0/(1+np.exp(diff_E)) 
        #r=np.exp(-diff_E) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x
#######    MAIN    ########
#Generate sample-dist
#J_max,J_min=5,0.0
#J_vec=np.random.uniform(J_min,J_max,d)
#J_mat=np.zeros((d,d))
#for i in range(d):
#    J_mat[i][(i+1)%d]=J_vec[i]*0.5
#    J_mat[(i+1)%d][i]=J_vec[i]*0.5


x = np.random.choice([-1,1],d)
#SAMPLING
for n in range(N_sample):
    for t in range(t_interval):
        x = np.copy(gen_mcmc(x))
    if(n==N_remove):X_sample = np.copy(x)
    elif(n>N_remove):X_sample=np.vstack((X_sample,np.copy(x)))
##########Calcuration of mean and variance###########
m=np.zeros(d)
n_bach=len(X_sample)
for i in range(d):
    m[i]=np.sum(X_sample.T[i])/n_bach
print(m)
XnXnt=np.zeros((d,d))
for n in range(n_bach):
    xn=np.matrix(X_sample[n])
    XnXnt=XnXnt+np.tensordot(xn,xn,axes=([0],[0]))/n_bach
mmt=np.tensordot(np.matrix(m),np.matrix(m),axes=([0],[0]))  
for i in range(d):XnXnt[i][i],mmt[i][i]=0.0,0.0
C=XnXnt-mmt

#########Calcuration of inverse values##########
#for l in range(d):
#    C[l][l]=0.0
#Cinv=inv(C)
#SVD type
U,s,V = np.linalg.svd(C, full_matrices=True)
sinv=1.0/s
Cinv=np.dot(V.T,np.dot(np.diag(sinv),U.T))

#########MFA#######
J_MF=-Cinv
#########MF+Onsager(TAP)##############
J_TAP=np.zeros((d,d))
for i in range(d):
    for j in range(i,d):
        print(i,j,"mi*mj=",m[i]*m[j],Cinv[i][j])
        J_TAP[i][j] = (np.sqrt(1.0-8*m[i]*m[j]*Cinv[i][j])-1.0 ) / (4.0*m[i]*m[j])
        J_TAP[j][i] = J_TAP[i][j]
#normalization
J_TAP_temp=np.copy(J_TAP)
for i in range(d):
    for j in range(i+1,d):
        J_TAP[i][j]=J_TAP_temp[i][j]/np.sqrt(J_TAP_temp[j][j]*J_TAP_temp[i][i])
        J_TAP[j][i]=J_TAP[i][j]

error_MF=np.sum(np.sum(np.abs(J_MF-J)))/(d*d)
error_TAP=np.sum(np.sum(np.abs(J_TAP-J)))/(d*d)

print("error_MF = ",error_MF)
print("error_TAP = ",error_TAP)
#visualize
plt.figure()
plt.subplot(231)
plt.imshow(J)
plt.colorbar()
plt.title("J")
plt.subplot(232)
plt.imshow(J_MF)
plt.colorbar()
plt.title("J_MF")
plt.subplot(233)
plt.imshow(J_TAP)
plt.colorbar()
plt.title("J_TAP")
plt.subplot(235)
plt.imshow(J_MF-J)
plt.colorbar()
plt.title("J_MF-J")
plt.subplot(236)
plt.imshow(J_TAP-J)
plt.colorbar()
plt.title("J_TAP-J")

plt.show()
