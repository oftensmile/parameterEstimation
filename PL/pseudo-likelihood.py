import numpy as np
import sys
import matplotlib.pyplot as plt
np.random.seed(0)

#d,N_sample_model,N_sample_data,N_remove=,128,1000,500
d,N_sample_model,N_sample_data,N_remove=64,600,600,200
t_interval=40
t_GD=400
beta=1.0
lr,eps=0.1,0.001
"""
J=np.random.choice([-1,1],(d,d))
for i in range(d):
    for j in range(i,d):
        J[j][i]=J[i][j]
for i in range(d):J[i][i]=0.0
"""
""" 
J=np.zeros((d,d))
for i in range(d):
    J[i][(i+1)%d]=1
    J[i][(i-1+d)%d]=1
"""

## TEST_DATA-> http://richardkwo.net/talks/InverseIsingSlides.pdf  ##
J=np.random.randn(d,d)/(np.sqrt(d**2))#d**2=#parameter
for i in range(d):
    for j in range(i+1,d):
        J[j][i]=J[i][j]
for i in range(d):J[i][i]=0.0

def gen_mcmc(x=[],theta=[[]]):
    #Heat Bath
    for i in range(d):
        diff_E=0.0
        diff_E=beta*x[i]*(np.dot(theta[i],x)+np.dot(theta.T[i],x))
        #r=1.0/(1+np.exp(2*diff_E)) 
        r=np.exp(-diff_E) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x

if __name__ == '__main__':
    #### SAMPLING ###
    mean=0.0
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

    #### Gradient Decent ###
    J_PL=np.random.rand(d,d)*0.5
    J_PL=J_PL.T+J_PL
    for i in range(d):J_PL[i][i]=0.0
        #Poseudo liklihood estimation
    len_sample_data=len(X_sample)
    for i in range(d):
        error_i_pre=1000
        for t in range(t_GD):
            grad_likelihood_i=np.zeros(d)
            for n in range(N_sample_data):
                xn=np.copy(X_sample[n])
                grad_likelihood_i=grad_likelihood_i + 2.0*xn*xn[i] / (1.0+np.exp(2.0*xn[i]*np.dot(J_PL[i],xn)))  /len_sample_data
                #
                #within=np.dot(J_PL[i],xn)
                #xn*(np.exp(2.0*xn[i]*within)/)
            #Added regularization term
            J_PL[i]=J_PL[i]+lr*grad_likelihood_i #-0.01 * np.sign(J_PL[i])
            ### This fulling zero is empirically necessary. ####
            J_PL[i][i]=0
            error_i=np.sum(np.abs(J_PL[i]-J[i]))/d
            print(i,t,error_i)
            ### Exit condition(Note ; This method can be used only witout stochastic process, which doesn't include mcmc calc.)
            if(error_i<eps or error_i_pre<error_i):
                break
            error_i_pre=error_i

    for i in range(d):J_PL[i][i]=0
    norm=np.sum(np.sum(J_PL))
    #J_PL=J_PL*0.0024/(1200.0*0.0028)#*(1.0/norm)
    #J_PL=J_PL*(1.0/norm)*(24.0/0.06)
    error_total=np.sqrt(np.sqrt(np.sum(np.sum(J-J_PL))))
    error_final=np.sum(np.sum(np.abs(J-J_PL/norm)))
    print("#final error=",error_final)
    """
    plt.figure()
    plt.subplot(141)
    plt.imshow(J ,interpolation='nearest')
    plt.colorbar()
    plt.title("J")
    plt.subplot(142)
    plt.imshow(J_PL, interpolation='nearest')
    plt.colorbar()
    plt.title("J_PL")
    plt.subplot(143)
    plt.imshow(J_PL-J, interpolation='nearest')
    plt.colorbar()
    plt.title("J_PL-J")
    plt.subplot(144)
    plt.imshow(C_data, interpolation='nearest')
    plt.colorbar()
    plt.title("C_data")
    plt.show()
    """
