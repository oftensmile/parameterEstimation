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
d, N_sample = 64,300 #124, 1000
N_remove = 100
#parameter ( MPF+GD )
lr,eps =0.1, 1.0e-100
t_gd_max=600 
beta=1.0
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

def claster_proposal(a=[]):
    m_a=-np.ones(d)
    r=1-np.exp(-2.0*beta)
    m=0
    ## To allocate a number on a cluster.
    max_m=1
    for i in range(d):
        u=np.random.uniform()
        if(a[i]!=a[(i+d-1)%d] or u>r):
            m+=1
            m_a[i]=m
            max_m=m
        else:
            m_a[i]=m

    ## Treatment for the P.B.C.
    for j in range(d):
        if(m_a[j]==0):
            m_a[j]=max_m
        else:
            break
    ## Propose single cluster update.
    sta=0
    for m in range(1,max_m+1):
        temp=np.copy(a)
        flag=0
        for i in range(sta,d):
            if(m_a[i]==m):
                flag=1
                temp[i]*=-1
            elif(m_a[i]!=m and flag==1):
                sta=i
                break
        if(m==max_m):
            for j in range(d):
                if(m_a[j]==m):
                    temp[j]*=-1
                else:
                    break
        if(m==1):
            proposal=np.copy(temp)
        else:
            proposal=np.vstack((proposal,np.copy(temp)))
    return proposal

#######    MAIN    ########
#Generate sample-dist
J_max,J_min=1.0,0.0
J_vec=np.random.uniform(J_min,J_max,d)
J_vec_sum=np.sum(J_vec)
x = np.random.uniform(-1,1,d)
x = np.array(np.sign(x))
name_result="result-Claster-dim"+str(d)+".dat"
fr=open(name_result,"w")
#SAMPLING
N_sample_list=[10,50,100,200,400,800]
for N_sample in N_sample_list:
    check=0
    time_s=time.time()
    name="Cluster-Update-dim"+str(d)+"-sample"+str(N_sample)+".dat"
    f=open(name,"w")
    for n in range(N_sample+N_remove):
        for t in range(t_interval):
            x = np.copy(gen_mcmc(J_vec,x))
        if(n==N_remove):X_sample = np.copy(x)
        elif(n>N_remove):X_sample=np.vstack((X_sample,np.copy(x)))
    #MPF
    #In this case I applied 
    theta_model=np.random.uniform(0,1,d)    #Initial guess
    init_theta=np.copy(theta_model)
    error_func=1000
    for t_gd in range(t_gd_max):
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
            ##  Cluster porposals ##
            set_of_proposal=claster_proposal(np.copy(x_nin))
            len_set_proposal=len(np.matrix(set_of_proposal))
            for l in range(len_set_proposal):
                if(len_set_proposal==1):
                    x_nin_l=np.copy(set_of_proposal)
                elif(len_set_proposal>1):
                    x_nin_l=np.copy(set_of_proposal[l])
                x_nin_l_shift=np.copy(x_nin_l[1:d])
                x_nin_l_shift=np.append(x_nin_l_shift,x_nin_l[0])
                x_nin_l_x_shift=x_nin_l*x_nin_l_shift
                E_nin_l=np.dot(x_nin_l_x_shift,theta_model)
                diff_E=E_nin_l-E_nin
                gradK_nin=gradK_nin-(x_nin_x_shift-x_nin_l_x_shift)*np.exp(0.5*diff_E)/len_set_proposal
            gradK=gradK+gradK_nin/n_bach
        theta_model=theta_model-lr*gradK
        sum_of_gradK=np.sum(np.sum(gradK))
        error_func=np.sqrt(np.sum((theta_model-J_vec)**2))/J_vec_sum
        f.write(str(t_gd)+" "+str(error_func)+"\n")
        if(error_prev<error_func):
            time_f=time.time()
            dt=time_f-time_s
            f.write("#time="+str(dt))
            f.close()
            fr.write(str(N_sample)+"  "+str(error_func)+" "+str(dt)+"\n")
            check=1
            break
    if(check==0):
        time_f=time.time()
        dt=time_f-time_s
        f.write("#time="+str(dt))
        f.close()
        fr.write(str(N_sample)+"  "+str(error_func)+" "+str(dt)+"\n")
fr.close()
#Plot
"""
bins=np.arange(1,d+1)
bar_width=0.2
plt.bar(bins,J_vec,color="blue",width=bar_width,label="true",align="center")
plt.bar(bins+bar_width,theta_model,color="red",width=bar_width,label="estimated",align="center")
plt.bar(bins+2*bar_width,init_theta,color="green",width=bar_width,label="initial",align="center")
plt.bar(bins+3*bar_width,gradK*10,color="gray",width=bar_width,label="gradK",align="center")
plt.legend()
filename="test_output_fixed6.png"
plt.savefig(filename)
plt.show()
"""
