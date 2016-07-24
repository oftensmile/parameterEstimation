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
#t_burn_emp, t_burn_model = 1100, 10#10000, 100
t_interval = 40
#parameter ( System )
#d, N_sample = 64,300 #124, 1000
d= 64
N_remove = 100
#parameter ( MPF+GD )
lr,eps =0.1, 1.0e-100
t_gd_max=800 
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
J_max,J_min=1.0,0.0
J_vec=np.random.uniform(J_min,J_max,d)
J_vec_sum=np.sqrt(np.sum(J_vec**2))
x = np.random.uniform(-1,1,d)
x = np.array(np.sign(x))
#name_tot="result-HMC-MFP-smaple"+"-d"+str(d)+".dat"
name_tot="result-MFP-smaple"+"-d"+str(d)+".dat"
f_tot=open(name_tot,"w")
#sample_list=[50,100,200,300,400,600,800]
sample_list=[10,30]
for N_sample in sample_list:
    #name="HMC-MFP-smaple"+str(N_sample)+"-d"+str(d)+".dat"
    name="MFP-smaple"+str(N_sample)+"-d"+str(d)+".dat"
    f=open(name,"w")
    #SAMPLING
    for n in range(N_sample+N_remove):
        for t in range(t_interval):
            x = np.copy(gen_mcmc(J_vec,x))
        if(n==N_remove):X_sample = np.copy(x)
        elif(n>N_remove):X_sample=np.vstack((X_sample,np.copy(x)))
    #MPF
    theta_model=np.random.uniform(0,1,d)    #Initial guess
    init_theta=np.copy(theta_model)
    error_func=1000
    flag=0
    for t_gd in range(t_gd_max):
        time_s=time.time()
        error_prev=error_func
        gradK=np.zeros(d)
        n_bach=len(X_sample)
        for nin in range(n_bach):
            x_nin=np.copy(X_sample[nin])
            gradK_nin=np.zeros(d)
            x_nin_shift=np.copy(x_nin[1:d])
            x_nin_shift=np.append(x_nin_shift,x_nin[0])
            x_nin_x_shift=x_nin*x_nin_shift
            E_nin=np.dot(x_nin_x_shift,theta_model)
            ##  propose direction of gradient of potential ## 
            grad_prop=np.zeros(d)
            cout_p=0
            for i in range(d):
                poten_direction=-(x[(i+1)%d]*theta_model[i]+x[(i+d-1)%d]*theta_model[(i+d-1)%d])
                if (x_nin[i]*poten_direction<0):
                    cout_p+=1
                    x_nin_p=np.copy(x_nin)
                    x_nin_p[i]*=-1
                    x_nin_p_shift=np.copy(x_nin_p[1:d])
                    x_nin_p_shift=np.append(x_nin_p_shift,x_nin_p[0])
                    x_nin_p_x_shift=x_nin_p*x_nin_p_shift
                    E_nin_p=np.dot(x_nin_p_x_shift,theta_model)
                    diff_E=E_nin_p-E_nin
                    grad_prop=grad_prop+(x_nin_x_shift-x_nin_p_x_shift)*np.exp(0.5*diff_E)
            gradK_nin=gradK_nin-grad_prop / cout_p
            for l1 in range(d):
                x_nin_l=np.copy(x_nin)
                x_nin_l[l1]*=-1
                x_nin_l_shift=np.copy(x_nin_l[1:d])
                x_nin_l_shift=np.append(x_nin_l_shift,x_nin_l[0])
                x_nin_l_x_shift=x_nin_l*x_nin_l_shift
                E_nin_l=np.dot(x_nin_l_x_shift,theta_model)
                diff_E=E_nin_l-E_nin
                gradK_nin=gradK_nin-(x_nin_x_shift-x_nin_l_x_shift)*np.exp(0.5*diff_E)/d
            
            gradK=gradK+gradK_nin/n_bach

        theta_model=theta_model-lr*gradK
        sum_of_gradK=np.sum(np.sum(gradK))
        error_func=np.sqrt(np.sum((theta_model-J_vec)**2))/J_vec_sum
        #print(t_gd,sum_of_gradK,error_func)
        f.write(str(t_gd)+" "+str(error_func)+"\n")
        if(error_prev<error_func):
            flag=1
            time_f=time.time()
            dt=time_f-time_s
            f_tot.write(str(N_sample)+" "+str(error_func)+" "+str(dt)+"\n")
            f.write("#:"+str(N_sample)+" "+str(error_func)+" "+str(dt)+"\n")
            f.close()
            break
    if(flag==0):
        time_f=time.time()
        dt=time_f-time_s
        f_tot.write(str(N_sample)+" "+str(error_func)+" "+str(dt)+"\n")
        f.write("#:"+str(N_sample)+" "+str(error_func)+" "+str(dt)+"\n")
        f.close()
f_tot.close()
