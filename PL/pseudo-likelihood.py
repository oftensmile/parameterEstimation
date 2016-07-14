import numpy as np
import sys
import matplotlib.pyplot as plt
np.random.seed(0)

d,N_sample_model,N_sample_data,N_remove=8,100,200,100
t_interval=40
t_GD=500
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
    x=np.random.choice([-1,1],d)
    for n in range(N_sample_data+N_remove):
        for t in range(t_interval):
            x=np.copy(gen_mcmc(x,J))
        if(n==N_remove):X_sample=np.copy(x)
        elif(n>N_remove):X_sample=np.vstack((X_sample,np.copy(x)))
    #### Data Covariance ###            
    """
    C_data=np.zeros((d,d))
    len_sample=len(X_sample)
    for n in range(len_sample):
        xn=np.matrix(np.copy(X_sample[n]))
        C_data=C_data+np.tensordot(xn,xn,axes=([0],[0]))/len_sample   
    for i in range(d):C_data[i][i]=0.0
    print("C_data=",C_data)    
    """
    #### Gradient Decent ###
    J_PL=np.random.rand(d,d)*0.5
    J_PL=J_PL.T+J_PL
    for i in range(d):J_PL[i][i]=0.0
        #Poseudo liklihood estimation
    len_sample_data=len(X_sample)
    for i in range(d):
        for t in range(t_GD):
            grad_likelihood_i=np.zeros(d)
            for n in range(N_sample_data):
                xn=np.copy(X_sample[n])
                grad_likelihood_i=grad_likelihood_i+xn*xn[i]*(1.0+np.exp(2.0*np.dot(J_PL[i],xn)))**(-1)*(1.0/len_sample_data)
            J_PL[i]=J_PL[i]+lr*grad_likelihood_i
            error_i=np.sum(np.abs(J_PL[i]-J[i]))/d
            print(i,t,error_i)
            if(error_i<eps):
                break
    print(J_PL)
