import numpy as np
from scipy import linalg 
import sys
import  matplotlib.pyplot as plt
np.random.seed(0)

d,N_sample_model,N_sample_data,N_remove=8,500,1000,300
t_interval=40
t_GD=1000
beta=1.0
lr,eps=0.1,0.001
J=np.random.choice([-1,1],(d,d))
for i in range(d):
    for j in range(i,d):
        if(j==i):J[i][i]==0
        else:J[j][i]=J[i][j]

def gen_mcmc(x=[],theta=[[]]):
    #Heat Bath
    for i in range(d):
        diff_E=0.0
        diff_E=beta*x[i]*(np.dot(theta[i],x)+np.dot(theta.T[i],x))
        r=1.0/(1+np.exp(diff_E)) 
        #r=np.exp(-diff_E) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x

if __name__ == '__main__':
    ### SAMPLING ###
    x=np.random.choice([-1,1],d)
    for n in range(N_sample_data+N_remove):
        for t in range(t_interval):
            x=np.copy(gen_mcmc(x,J))
        if(n==N_remove):X_sample=np.copy(x)
        elif(n>N_remove):X_sample=np.vstack((X_sample,np.copy(x)))
    #### Data Covariance ###            
    C_data=np.zeros((d,d))
    len_sample=len(X_sample)
    for n in range(len_sample):
        xn=np.matrix(np.copy(X_sample[n]))
        C_data=C_data+np.tensordot(xn,xn,axes=([0],[0]))/len_sample   
    for i in range(d):C_data[i][i]=0.0
    print("C_data=",C_data)    
    #### Gradient Decent ###
    J_BF=np.random.rand(d,d)*0.5
    J_BF=J_BF.T+J_BF
    for i in range(d):J_BF[i][i]=0.0
    for t in range(t_GD):
        ### mcmc-mean of correlation ###
        x_model=np.copy(X_sample[0])  #using CD
        for n in range(N_sample_model+N_remove):
            for t_mcmc in range(t_interval):
                x_model=np.copy(gen_mcmc(x_model,J_BF))
            if(n==N_remove):X_sample_model=np.copy(x_model)
            elif(n>N_remove):X_sample_model=np.vstack((X_sample_model,np.copy(x_model)))
        len_samp_model=len(X_sample_model)
        C_model=np.zeros((d,d))
        for n in range(len_samp_model):
            xn_model=np.matrix(np.copy(X_sample_model[n]))
            C_model=C_model+np.tensordot(xn_model,xn_model,axes=([0],[0]))/len_samp_model
        for i in range(d):C_model[i][i]=0.0
        temp_J_BF=np.copy(J_BF)
        J_BF=temp_J_BF+lr*(C_data-C_model)
        delta=np.sum(np.sum(np.abs(C_data-C_model)))
        #error_BF=np.sum(np.sum(np.abs(J_BF-temp_J_BF)))/d**2
        error_BF=np.sum(np.sum(np.abs(J_BF-J)))/d**2
        #print("#delta=",t,delta)
        print("#error=",t,error_BF)
        if(error_BF<eps):
            break
    print(J_BF)

    plt.figure()
    plt.subplot(131)
    plt.imshow(J)
    plt.colorbar()
    plt.title("J")
    plt.subplot(132)
    plt.imshow(J_BF)
    plt.colorbar()
    plt.title("J_BF")
    plt.subplot(133)
    plt.imshow(J_BF-J)
    plt.colorbar()
    plt.title("J_BF-J")
    plt.show()


