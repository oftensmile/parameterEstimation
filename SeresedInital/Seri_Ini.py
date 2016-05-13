import numpy as np
from scipy import linalg 
import matplotlib.pyplot as plt
import time
np.random.seed(0)
d,T,N,heatN,J=16,300,1000,20,1
burn_t_long, burn_t_short = 1000, 100
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
#put into previoust equilibrium state as an initial state in mcmc
def sum_xixj(mc_step,n_sample,theta=[[]],pre_y=[]):
    xixj=np.zeros((d,d))
    if(mc_step==0):
        y=np.ones(d)
        y=gen_mcmc(burn_t_long,y,theta)
    else:
        y=np.copy(pre_y)
        y=gen_mcmc(burn_t_short,y,theta)
    for n in range(n_sample):
        y=gen_mcmc(1,y,theta)
        xixj=xixj+np.tensordot(y,y,axes=([0][0]))/n_sample
    for l in range(d):xixj[l][l]=0
    return xixj,y

#################### MAIN #########################
Y=np.ones(d)
dl_sample,Y =sum_xixj(0,N,theta,Y)
# initial theta
theta_est=0.1*np.random.rand(d,d)
theta_est=0.5*(theta_est + theta_est.T)
loss,delta=np.zeros(T),np.zeros(T)
for l in range(d):theta_est[l][l]=0
#using AdaGrad
r,a,epc=0,1,0.001
y=np.ones(d)
time_st=time.time()
for k in range(T):
    dl_model,y=sum_xixj(k,heatN,theta_est,y)
    #r+=np.sum(dl_model*dl_model)
    #lr=a/(np.sqrt(r)+epc)
    lr=1.0/np.log(2+k)
    grad=dl_sample - dl_model
    theta_est=theta_est-lr*grad
    loss[k]=np.absolute(theta-theta_est).sum()
    print(loss[k])
    delta[k]=np.absolute(grad).sum()
time_en=time.time()
result=[gen_mcmc(1000,np.ones(d),theta_est)]
#print("theta_est=",theta_est)
print("#working time = ",time_en-time_st)
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
plt.subplot(325)
plt.imshow(result)
plt.title('generated config')
plt.show()
