import numpy as np
from scipy import linalg 
import numpy.matlib
import matplotlib.pyplot as plt
from scipy import linalg
from mpl_toolkits.mplot3d.axes3d import Axes3D
np.set_printoptions(precision=3)
np.random.seed(0)
d,T,N,heatN,J=8,1000,1000,20,1
h,lam=1.0**2,0.1
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
        xixj=xixj+np.tensordot(y,y,axes=([0][0]))
    for l in range(d):xixj[l][l]=0
    return xixj/n_sample

#sampling of the theta matrix
def gen_theta(t_wait,samp_theta=[[]],X=[[]]):
    for t in range(t_wait):
        for i in range(d):
            for j in range(i+1,d):
                dtheta=0.1*np.random.uniform(-1,1)# 
                value=J*dtheta*X[i][j]
                if(value>0):
                    r=np.exp(-value)
                    R=np.random.uniform(0,1)
                    if(r<R):
                        samp_theta[i][j]=samp_theta[i][j]-dtheta
                        samp_theta[j][i]=samp_theta[j][i]-dtheta
    return samp_theta


def kernel_regres(x,target):
    xx=np.dot(np.matrix(x),np.matrix(x).T)
    x2=np.diagonal(xx)
    x2=np.matlib.repmat(np.matrix(x2).T,1,len(x))+np.matlib.repmat(np.matrix(x2),len(x),1)
    K=np.exp(-(x2-2.0*xx)/h)
    Kinv=np.linalg.inv(K+lam*np.diag(np.ones(len(x))))#+lam*np.diag(np.ones(len(x)))
    return np.dot(Kinv,target)

def k_of_x(n_sample,x,sample):
    kofx=np.zeros(n_sample)
    for i in range(n_sample):
        kofx[i]=np.exp(-(np.dot(sample[:,i].T,sample[:,i])+np.dot(x,x)-2*np.dot(x,sample[:,i]))/h)
    return kofx
#################### MAIN #########################
dl_sample=sum_xixj(N,theta)
theta_est=0.1*np.random.rand(d,d)
theta_est=0.5*(theta_est + theta_est.T)
loss,loss_mcmc,delta=np.zeros(T),np.zeros(T),np.zeros(T)
samp_theta=np.ones((d,d),dtype=np.float)
r,a,epc=0.0,2.0,0.001
for t in range(T):
    #SGD-method
    #dl_model=sum_xixj(heatN,theta_est)
    #r+=np.sum(dl_model*dl_model)
    #lr=a/(np.sqrt(r)+epc)
    #grad=dl_sample-dl_model
    #theta_est=theta_est-lr*grad
    norm_theta=np.sum(abs(np.array(theta)))
    #MCMC-method
    samp_theta=gen_theta(10,samp_theta,dl_sample)
    if(t==4):print(np.shape(samp_theta))
    #loss[t]=np.absolute(theta/norm_theta-theta_est/np.sum(abs(np.array(theta_est)))).sum()
    loss_mcmc[t]=np.absolute(theta/norm_theta-samp_theta/np.sum(abs(np.array(samp_theta)))).sum()
    #loss[t]=np.absolute(theta-theta_est).sum()
    #loss_mcmc[t]=np.absolute(theta-samp_theta).sum()
    #delta[t]=np.absolute(grad).sum()

step=np.arange(T)
sgd,=plt.plot(step,loss,'r-',label='SGD')
mcs,=plt.plot(step,loss_mcmc,'b-',label='mcmc')
plt.legend([sgd,mcs],['SGD','mcmc'])
plt.show()
plt.subplot(131)
plt.imshow(theta)
plt.colorbar()
plt.title('ture')
plt.subplot(132)
plt.imshow(theta_est)
plt.colorbar()
plt.title('SGD')
plt.subplot(133)
plt.imshow(samp_theta)
plt.colorbar()
plt.title('mcmc')
plt.show()


