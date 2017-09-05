import numpy as np
from scipy import linalg 
import matplotlib.pyplot as plt
np.random.seed(0)
d,T,N,heatN,J=8,800,1000,20,1
theta=[[1 if i==(j+1+d)%d or i==(j-1+d)%d else 0 for i in range(d)] for j in range(d)]
def gen_mcmc(t_wait, x=[],theta=[[]]):
    for t in range(t_wait):
        for i in range(d):
            valu=J*(np.dot(theta[:][i],x)-x[i]*theta[i][i])
            r=np.exp(-valu)/(np.exp(-valu)+np.exp(valu))
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
    for n in range(n_sample):
        y=gen_mcmc(3,y,theta)
        xixj=xixj+np.tensordot(y,y,axes=([0][0]))/n_sample
    for l in range(d):xixj[l][l]=0
    return xixj

#################### MAIN #########################
dl_sample=sum_xixj(N,theta)
# initial theta
theta_est=0.1*np.random.rand(d,d)
theta_est=0.5*(theta_est + theta_est.T)
loss=np.zeros(T)
for l in range(d):theta_est[l][l]=0
for k in range(T):
    lr=0.3/np.log(k+2.0)
    dl_model=sum_xixj(heatN,theta_est)
    theta_est=theta_est-lr*(dl_sample - dl_model)
    loss[k]=np.absolute(theta-theta_est).sum()

result=[gen_mcmc(1000,np.ones(d),theta_est)]
print("theta_est=",theta_est)
plt.subplot(221)
plt.imshow(theta)
plt.title('true theta')
plt.subplot(222)
plt.imshow(theta_est)
plt.title('estimated theta ')
plt.subplot(223)
plt.imshow(result)
plt.title('generated config')
plt.subplot(224)
plt.plot(loss)
plt.title('loss function')
plt.show()
