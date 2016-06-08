#2016/05/19
##############
#   H = J*sum(xixj), J in R^1
##############
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
import csv
np.random.seed(0)
#parameter ( MCMC )
t_interval = 40
#parameter ( System )
d_x,d_y, N_sample = 4,4,240 #124, 1000
N_remove=40
#parameter ( MPF+GD )
lr,eps =0.1, 1.0e-100
t_gd_max=200 
def gen_mcmc(x=[],J=[[]] ):
    for ix in range(d_x):
        for iy in range(d_y):
            #Heat Bath
            diff_E=-2.0*x[ix+iy*d_x]*( J[0][(ix+d_x-1)%d_x+iy*d_x]*x[(ix+d_x-1)%d_x+iy*d_x]+J[0][ix+iy*d_x]*x[(ix+1)%d_x+iy*d_x] 
                    + J[1][ix+d_x*((iy+d_y-1)%d_y)]*(x[ix+d_x*((iy+d_y-1)%d_y)]+J[1][ix+d_x*iy]*x[ix+d_x*((iy+1)%d_y)]) )#E_new-E_old
            r=1.0/(1+np.exp(diff_E)) 
            R=np.random.uniform(0,1)
            if(R<=r):
                x[ix+iy*d_x]=x[ix+iy*d_x]*(-1)
    return x

#######    MAIN    ########
#Generate sample-dis
J = np.random.uniform(0,2,d_x*d_y)
J= np.vstack((J,np.random.uniform(0,2,d_x*d_y)))    #J1=[],J2=[]
print("shape of J[0]=",np.shape(J[1]))
x = np.random.uniform(-1,1,d_x*d_y)
x = np.array(np.sign(x))
#SAMPLING
for n in range(N_sample):
    for t in range(t_interval):
        x = np.copy(gen_mcmc(x,J))
        if(n==N_remove):X_sample = np.copy(x)
        elif(n>N_remove):X_sample=np.vstack((X_sample,np.copy(x)))
#MPF
theta_model = np.random.uniform(2,4,d_x*d_y)
theta_model = np.vstack((theta_model,np.random.uniform(0,2,d_x*d_y)))#J1=[],J2=[]
print("size of theta_model[0] = ",np.shape(theta_model[0]))
print("#diff_E diff_E1_nin diff_E2_nin")
for t_gd in range(t_gd_max):
    gradK1=np.zeros(d_x*d_y)
    gradK2=np.zeros(d_y*d_y)
    n_bach=len(X_sample)
    for nin in range(n_bach):
        x_nin=np.copy(X_sample[nin])
        gradK1_nin=np.zeros(d_x*d_y)
        gradK2_nin=np.zeros(d_y*d_x)
        for ix in range(d_x):
            for iy in range(d_y):
                #diff_E=E(x_new)-E(x_old)
                diff_delE1_p_nin=x_nin[ix+iy*d_x]*x_nin[(ix+1)%d_x+iy*d_x]
                diff_delE1_m_nin=x_nin[ix+iy*d_x]*x_nin[(ix+d_x-1)%d_x+iy*d_x]
                diff_delE2_p_nin=x_nin[ix+iy*d_x]*x_nin[ix+d_x*((iy+1)%d_y)] 
                diff_delE2_m_nin=x_nin[ix+iy*d_x]*x_nin[ix+d_x*((iy+d_y-1)%d_y)] 
                diff_E1_nin=J[0][(ix+1)%d_x+iy*d_x]*diff_delE1_p_nin+J[0][ix+iy*d_x]*diff_delE1_m_nin
                diff_E2_nin=J[1][ix+d_x*((iy+1)%d_y)]*diff_delE2_p_nin+J[1][ix+iy*d_x]*diff_delE2_m_nin
                diff_E_nin=diff_E1_nin+diff_E2_nin
                gradK1_nin[(ix+1)%d_x+iy*d_x]+=diff_delE1_p_nin*np.exp(diff_E_nin)#/(d_x*d_y)
                gradK1_nin[ix+iy*d_x]+=diff_delE1_m_nin*np.exp(diff_E_nin)#/(d_x*d_y)
                gradK2_nin[ix+((iy+1)%d_y)*d_x]+=diff_delE2_p_nin*np.exp(diff_E_nin)#/(d_x*d_y)
                gradK2_nin[ix+iy*d_x]+=diff_delE2_m_nin*np.exp(diff_E_nin)#/(d_x*d_y)
        gradK1=gradK1+gradK1_nin/n_bach
        gradK2=gradK2+gradK2_nin/n_bach
        #print("#gradK1[0]=",gradK1[0], "gradK2=",gradK2[0])
    
    theta_model[0]=theta_model[0] - lr * gradK1
    theta_model[1]=theta_model[1] - lr * gradK2
    theta_diff1=np.sum(abs(theta_model[0]-J[0]))/(d_x*d_y)
    theta_diff2=np.sum(abs(theta_model[1]-J[1]))/(d_x*d_y)
    print(t_gd,np.sum(np.abs(gradK1)),np.sum(np.abs(gradK2)),theta_diff1,theta_diff2)
#print("#theta1,theta2 (true)=",J1,J2,"theta1,theta2 _estimated=",theta_model1,theta_model2)
