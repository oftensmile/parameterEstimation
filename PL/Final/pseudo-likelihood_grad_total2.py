import numpy as np
import sys
import matplotlib.pyplot as plt
np.random.seed(0)

#d,N_sample_model,N_sample_data,N_remove=8,100,2000,1000
d,N_sample_model,N_sample_data,N_remove=16,600,400,100
t_interval=40
t_GD=100
beta=1.0
lr,eps=1.0,0.0001
 
J_min,J_max=0.0,1.0
J_vec=np.random.uniform(J_min,J_max,d)
J_mat=np.zeros((d,d))
for i in range(d):
    J_mat[i][(i+1)%d]=J_vec[i]*0.5
    J_mat[(i+1)%d][i]=J_vec[i]*0.5
J_norm=np.sqrt(np.sum(np.sum(J_mat**2)))
#def gen_mcmc(x=[],theta=[[]]):
def gen_mcmc(x=[],J=[]):
    #Heat Bath
    for i in range(d):
        diff_E=0.0
        #diff_E=beta*x[i]*(np.dot(theta[i],x)+np.dot(theta.T[i],x))
        diff_E=beta*2.0*x[i]*(J[i]*x[(i+1)%d]+J[(i+d-1)%d]*x[(i+d-1)%d])
        #r=1.0/(1+np.exp(2*diff_E)) 
        r=np.exp(-diff_E) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x

if __name__ == '__main__':
    #### SAMPLING ###
    x=np.random.choice([-1,1],d)
    for n in range(N_sample_data+N_remove):
        for t in range(t_interval):
            #x=np.copy(gen_mcmc(x,J_mat))
            x=np.copy(gen_mcmc(x,J_vec))
        if(n==N_remove):X_sample=np.copy(x)
        elif(n>N_remove):X_sample=np.vstack((X_sample,np.copy(x)))
    #### Data Covariance ###
    C_data=np.zeros((d,d))
    len_sample=len(X_sample)
    for n in range(len_sample):
        xn=np.matrix(np.copy(X_sample[n]))
        C_data=C_data+np.tensordot(xn,xn,axes=([0],[0]))/len_sample   
    #for i in range(d):C_data[i][i]=0.0

    #### Gradient Decent ###
    J_PL=np.random.rand(d,d)*0.5
    J_PL=J_PL.T+J_PL
    for i in range(d):J_PL[i][i]=0.0
        #Poseudo liklihood estimation
    len_sample_data=len(X_sample)

    error_pre=1000
    for t in range(t_GD):
        grad_likelihood=np.zeros((d,d))
        for i in range(d):
            for n in range(N_sample_data):
                xn=np.copy(X_sample[n])
                grad_likelihood[i]=grad_likelihood[i] + 2.0*xn*xn[i] / (1.0+np.exp(2.0*xn[i]*np.dot(J_PL[i],xn)))  /len_sample_data
        #Added regularization term
        for i in range(d): grad_likelihood[i][i]=0.0
        grad_likelihood = 0.5 * ( np.copy(grad_likelihood)+np.copy(grad_likelihood).T )
        J_PL=np.copy(J_PL)+lr*grad_likelihood -0.02 * np.sign(J_PL)
        #J_PL=0.5*(  np.copy(J_PL) + np.copy(J_PL.T) )
        J_PL_norm=np.sqrt(np.sum(np.sum(J_PL**2)))
        J_PL=np.copy(J_PL)*(J_norm/J_PL_norm)
        ### This fulling zero is empirically necessary. ####
        #for s in range(d):J_PL[s][s]=0.0
        error=np.sqrt(np.sum(np.sum((J_PL-J_mat)**2)))/J_norm
        print(t,error)
        ### Exit condition(Note ; This method can be used only witout stochastic process, which doesn't include mcmc calc.)
        #if(error<eps or error_pre<error):
        if(error<eps ):
            break
        error_pre=error
    plt.figure()
    plt.subplot(141)
    plt.imshow(J_mat ,interpolation='nearest')
    plt.colorbar()
    plt.title("J_true")
    plt.subplot(142)
    plt.imshow(J_PL, interpolation='nearest')
    plt.colorbar()
    plt.title("J_PL")
    plt.subplot(143)
    plt.imshow(J_PL-J_mat, interpolation='nearest')
    plt.colorbar()
    plt.title("J_PL-J")
    plt.subplot(144)
    plt.imshow(C_data, interpolation='nearest')
    plt.colorbar()
    plt.title("C_data")
    plt.show()
