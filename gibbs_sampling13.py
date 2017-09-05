import numpy as np
from scipy import linalg 
import matplotlib.pyplot as plt
np.random.seed(0)
d,T,N,heatN,J=16,1000,1000,20,10.0
theta=[[1 if i==(j+1+d)%d or i==(j-1+d)%d else 0 for i in range(d)] for j in range(d)]
def gen_mcmc(t_wait, x=[],theta=[[]]):
    for t in range(t_wait):
        for i in range(d):
            valu=J*((np.dot(theta[:][i],x)-x[i]*theta[i][i]))
            #r=np.exp(-valu)/(np.exp(-valu)+np.exp(valu))
            r=np.exp(-valu)
            R=np.random.uniform(0,1)
            if(R<=r):
                x[i]=1
            else:
                x[i]=-1
    return x

def sum_xixj(n_sample,theta=[[]]):
    xixj=np.zeros((d,d))
    y=np.ones(d)
    y=gen_mcmc(500,y,theta)
    correlation=0
    x_previous=y
    for n in range(n_sample):
        y=gen_mcmc(10,y,theta)
        xixj=xixj+np.tensordot(y,y,axes=([0][0]))/n_sample
        #below 3-lines are just for check
        correlation=correlation + 1.0*y*x_previous
    for l in range(d):xixj[l][l]=0
    #print(correlation/n_sample)
    return xixj

#################### MAIN #########################
dl_sample=sum_xixj(N,theta)
# initial theta
theta_est=0.1*np.random.rand(d,d)
theta_est=0.5*(theta_est + theta_est.T)
loss,delta=np.zeros(T),np.zeros(T)
dl1=np.zeros((d,d))
for l in range(d):theta_est[l][l]=0
for k in range(T):
    dl_model=sum_xixj(heatN,theta_est)
    lr=0.3/np.log(k+2.0)
    for i in range(d):
        for j in range(d):
            dl1[i][j]=theta_est[i][j]/(np.abs(theta_est[i][j])+0.0000000001)
    theta_est=theta_est-lr*(dl_sample - dl_model +0.01*dl1)
    loss[k]=np.absolute(theta-theta_est).sum()
    grad=dl_sample-dl_model
    delta[k]=np.absolute(grad).sum()
#print("theta_est=",theta_est)
plt.subplot(321)
plt.imshow(theta)
plt.colorbar()
plt.title('true theta')
plt.subplot(322)
plt.imshow(theta_est)
plt.colorbar()
plt.title('estimated theta ')
plt.subplot(323)
plt.plot(loss)
plt.title('loss function')
plt.subplot(324)
plt.plot(delta)
plt.title('grad')
plt.show()
