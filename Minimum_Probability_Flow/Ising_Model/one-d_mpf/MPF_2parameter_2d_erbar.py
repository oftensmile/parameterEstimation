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
d_x,d_y, N_sample = 4,4,140 #124, 1000
N_remove=40
#parameter ( MPF+GD )
lr,eps =0.1, 1.0e-100
t_gd_max=200 
def gen_mcmc(J1,J2,x=[] ):
    for ix in range(d_x):
        for iy in range(d_y):
            #Heat Bath
            diff_E=-2.0*x[ix+iy*d_x]*( J1*(x[(ix+d_x-1)%d_x+iy*d_x]+x[(ix+1)%d_x+iy*d_x]) + J2* (x[ix+d_x*((iy+d_y-1)%d_y)]+x[ix+d_x*((iy+1)%d_y)]) )#E_new-E_old
            r=1.0/(1+np.exp(diff_E)) 
            R=np.random.uniform(0,1)
            if(R<=r):
                x[ix+iy*d_x]=x[ix+iy*d_x]*(-1)
    return x

#######    MAIN    ########
#Generate sample-dist
J1,J2=2.0,1.0 # =theta_sample
x = np.random.uniform(-1,1,d_x*d_y)
x = np.array(np.sign(x))
#SAMPLING
for n in range(N_sample):
    for t in range(t_interval):
        x = np.copy(gen_mcmc(J1,J2,x))
        if(n==N_remove):X_sample = np.copy(x)
        elif(n>N_remove):X_sample=np.vstack((X_sample,np.copy(x)))
#MPF
theta_model1,theta_model2=1.5, 1.5  #Initial Guess
print("#diff_E diff_E1_nin diff_E2_nin")
for t_gd in range(t_gd_max):
    gradK1,gradK2=0.0,0.0
    e_bar1,e_bar2=0.0,0.0
    n_bach=len(X_sample)
    for nin in range(n_bach):
        x_nin=np.copy(X_sample[nin])
        gradK1_nin,gradK2_nin=0.0,0.0
        for ix in range(d_x):
            for iy in range(d_y):
                #diff_E=E(x_new)-E(x_old)
                diff_delE1_nin=x_nin[ix+iy*d_x]*(x[(ix+d_x-1)%d_x+iy*d_x]+x[(ix+1)%d_x+iy*d_x])
                diff_delE2_nin=x_nin[ix+iy*d_x]*(x[ix+d_x*((iy+d_y-1)%d_y)]+x[ix+d_x*((iy+1)%d_y)])
                diff_E1_nin=diff_delE1_nin*theta_model1
                diff_E2_nin=diff_delE2_nin*theta_model2
                diff_E_nin=diff_E1_nin+diff_E2_nin
                gradK1_nin+=diff_delE1_nin*np.exp(diff_E_nin)/(d_x*d_y)
                gradK2_nin+=diff_delE2_nin*np.exp(diff_E_nin)/(d_x*d_y)
        gradK1+=gradK1_nin/n_bach
        gradK2+=gradK2_nin/n_bach
    theta_model1=theta_model1 - lr * gradK1
    theta_model2=theta_model2 - lr * gradK2
    theta_diff1=abs(theta_model1-J1)
    theta_diff2=abs(theta_model2-J2)
    #Error Bar
    for nin in range(n_bach):
        e_bar1_nin,e_bar2_nin=0.0,0.0
        for ix in range(d_x):
            for iy in range(d_y):
                diff_delE1_nin=x_nin[ix+iy*d_x]*(x[(ix+d_x-1)%d_x+iy*d_x]+x[(ix+1)%d_x+iy*d_x])
                diff_delE2_nin=x_nin[ix+iy*d_x]*(x[ix+d_x*((iy+d_y-1)%d_y)]+x[ix+d_x*((iy+1)%d_y)])
                diff_E1_nin=diff_delE1_nin*theta_model1
                diff_E2_nin=diff_delE2_nin*theta_model2
                diff_E_nin=diff_E1_nin+diff_E2_nin
                e_bar1_nin+=(diff_delE1_nin**2)*np.exp(diff_E_nin)/(d_x*d_y)
                e_bar2_nin+=(diff_delE2_nin**2)*np.exp(diff_E_nin)/(d_x*d_y)
        e_bar1+=e_bar1_nin/n_bach
        e_bar2+=e_bar2_nin/n_bach
    print(t_gd,np.abs(gradK1),np.abs(gradK2),theta_diff1,theta_diff2,np.sqrt(e_bar1),np.sqrt(e_bar2))
print("#theta1,theta2 (true)=",J1,J2,"theta1,theta2 _estimated=",theta_model1,theta_model2)
