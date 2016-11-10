#2016/05/19
##############
#   H = -J*sum(xixj), J in R^1
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
d_x,d_y, N_sample = 1,5,200 #124, 1000
N_remove=100
#parameter ( MPF+GD )
lr,eps =0.01, 1.0e-100
t_gd_max=100 
def gen_mcmc(x=[],J=[[]] ):
    for ix in range(d_x):
        for iy in range(d_y):
            #Heat Bath
            diff_E=2.0*x[ix+iy*d_x]*( 
                    J[0][ix+iy*d_x]*x[(ix+1)%d_x+iy*d_x]+ 
                    J[0][(ix+d_x-1)%d_x+iy*d_x]*x[(ix+d_x-1)%d_x+iy*d_x]+
                    J[1][ix+iy*d_x]*x[ix+d_x*((iy+1)%d_y)]+
                    J[1][ix+d_x*((iy+d_y-1)%d_y)]*x[ix+d_x*((iy+d_y-1)%d_y)])#E_new-E_old
            #r=1.0/(1+np.exp(-diff_E)) 
            r=np.exp(diff_E)
            R=np.random.uniform(0,1)
            if(R<=r):
                x[ix+iy*d_x]=x[ix+iy*d_x]*(-1)
    return x

#######    MAIN    ########
#Generate sample-dis
#Jmin, Jmax=0.0,1.0
Jmin, Jmax=-0.01,0.01
J = np.random.uniform(Jmin,Jmax,d_x*d_y)
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
    gradK2=np.zeros(d_x*d_y)
    n_bach=len(X_sample)
    for nin in range(n_bach):
        x_nin=np.copy(X_sample[nin])
        gradK1_nin=np.zeros(d_x*d_y)
        gradK2_nin=np.zeros(d_x*d_y)
        for ix in range(d_x):
            for iy in range(d_y):
                x_y_xp_y=x_nin[ix+iy*d_x]*x_nin[(ix+1)%d_x+iy*d_x]
                xp_y_xpp_y=x_nin[(ix+1)%d_x+iy*d_x]*x_nin[(ix+2)%d_x+iy*d_x]
                xm_y_x_y=x_nin[ix+iy*d_x]*x_nin[(ix+d_x-1)%d_x+iy*d_x]
                xp_y_xp_yp=x_nin[(ix+1)%d_x+iy*d_x]*x_nin[(ix+1)%d_x+((iy+1)%d_y)*d_x]
                xp_y_xp_ym=x_nin[(ix+1)%d_x+iy*d_x]*x_nin[(ix+1)%d_x+((iy+d_y-1)%d_y)*d_x]
                x_y_x_yp=x_nin[ix+iy*d_x]*x_nin[ix+((iy+1)%d_y)*d_x]
                x_yp_xp_yp=x_nin[ix+((iy+1)%d_y)*d_x]*x_nin[(ix+1)%d_x+((iy+1)%d_y)*d_x]
                x_yp_x_ypp=x_nin[ix+((iy+1)%d_y)*d_x]*x_nin[ix+((iy+2)%d_y)*d_x]
                xm_yp_x_yp=x_nin[(ix+d_x-1)%d_x+((iy+1)%d_y)*d_x]*x_nin[ix+((iy+1)%d_y)*d_x]
                x_y_x_ym=x_nin[ix+iy*d_x]*x_nin[ix+((iy+d_y-1)%d_y)*d_x]
                xm_y_xm_yp=x_nin[(ix+d_x-1)%d_x+iy*d_x]*x_nin[(ix+d_x-1)%d_x+((iy+1)%d_y)*d_x]
                x_ym_xp_ym=x_nin[ix+((iy+d_y-1)%d_y)*d_x]*x_nin[(ix+1)%d_x+((iy+d_y-1)%d_y)*d_x]
                t1_x_y=theta_model[0][ix+iy*d_x]
                t2_x_y=theta_model[1][ix+iy*d_x]
                t1_xm_y=theta_model[0][(ix+d_x-1)%d_x+iy*d_x]
                t2_x_ym=theta_model[1][ix+((iy+d_y-1)%d_y)*d_x]
                t1_xp_y=theta_model[0][(ix+1)%d_x+iy*d_x]
                t2_xp_y=theta_model[1][(ix+1)%d_x+iy*d_x]
                t1_xm_yp=theta_model[0][(ix+d_x-1)%d_x+((iy+1)%d_y)*d_x]#ix_iym ?
                t2_xp_ym=theta_model[1][(ix+1)%d_x+((iy+d_y-1)%d_y)*d_x]#ixm_iy ?
                t1_x_yp=theta_model[0][ix+((iy+1)%d_y)*d_x]
                t2_x_yp=theta_model[1][ix+((iy+1)%d_y)*d_x]
                t2_xm_y=theta_model[1][(ix+d_x-1)%d_x+iy*d_x]
                t1_x_ym=theta_model[0][ix+((iy+d_y-1)%d_y)*d_x]
                #Gradient Decent(=-) or Accent(=+) ?
                A0=-(x_y_xp_y*t1_x_y + x_y_x_yp*t2_x_y + xm_y_x_y*t1_xm_y + x_y_x_ym*t2_x_ym)
                A1=-(xp_y_xpp_y*t1_xp_y + xp_y_xp_yp*t2_xp_y + x_y_xp_y*t1_x_y + xp_y_xp_ym*t2_xp_ym)
                A2=-(x_yp_xp_yp*t1_x_yp + x_yp_x_ypp*t2_x_yp + xm_yp_x_yp*t1_xm_yp + x_y_x_yp*t2_x_y)
                gradK1_nin[ix+iy*d_x]+=x_y_xp_y * (
                np.exp(-(x_y_xp_y*t1_x_y + x_y_x_yp*t2_x_y + xm_y_x_y*t1_xm_y + x_y_x_ym*t2_x_ym)) +
                np.exp(-(xp_y_xpp_y*t1_xp_y + xp_y_xp_yp*t2_xp_y + x_y_xp_y*t1_x_y + xp_y_xp_ym*t2_xp_ym)) )*(1.0/(d_x*d_y))
                #np.exp(-(xp_y_xpp_y*t1_xp_y + xp_y_xp_yp*t2_xp_y + x_y_xp_y*t1_x_y + xm_y_xm_yp*t2_xm_y)) )*(1.0/(d_x*d_y))

                gradK2_nin[ix+iy*d_x]+=x_y_x_yp * (
                np.exp(-(x_y_xp_y*t1_x_y + x_y_x_yp*t2_x_y + xm_y_x_y*t1_xm_y + x_y_x_ym*t2_x_ym)) +
                np.exp(-(x_yp_xp_yp*t1_x_yp + x_yp_x_ypp*t2_x_yp + xm_yp_x_yp*t1_xm_yp + x_y_x_yp*t2_x_y)) )*(1.0/(d_x*d_y))
                #np.exp(-(x_yp_xp_yp*t1_x_yp + x_yp_x_ypp*t2_x_yp + x_ym_xp_ym*t1_x_ym + x_y_x_yp*t2_x_y)) )*(1.0/(d_x*d_y))
        
        gradK1=gradK1+gradK1_nin/n_bach
        gradK2=gradK2+gradK2_nin/n_bach
    
    #print("A0=",A0,"A1=",A1,"A2=",A2)
    theta_model[0]=theta_model[0] - lr * gradK1
    theta_model[1]=theta_model[1] - lr * gradK2
    theta_diff1=np.sum(abs(theta_model[0]-J[0]))#/(d_x*d_y)
    theta_diff2=np.sum(abs(theta_model[1]-J[1]))#/(d_x*d_y)
    print(t_gd,np.sum(np.abs(gradK1)),np.sum(np.abs(gradK2)),theta_diff1,theta_diff2)
#print("#theta1,theta2 (true)=",J1,J2,"theta1,theta2 _estimated=",theta_model1,theta_model2)
