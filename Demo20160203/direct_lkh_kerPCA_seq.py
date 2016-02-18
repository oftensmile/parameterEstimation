import numpy as np
from scipy import linalg 
import numpy.matlib
import matplotlib.pyplot as plt
from scipy import linalg
from mpl_toolkits.mplot3d.axes3d import Axes3D
np.set_printoptions(precision=3)
np.random.seed(0)
d,T,N,heatN,J=3,1000,1000,20,1
h,lam=10.0,0.1
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
                dtheta=0.1*np.random.uniform(-1,1)#theta(t)-theta(t+1)
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
# initial theta
theta_est=np.random.rand(d,d)
theta_est=0.5*(theta_est + theta_est.T)
loss=np.zeros(T)

samp_theta=np.ones((d,d),dtype=np.float)
samp_theta=gen_theta(800,samp_theta,dl_sample)
phi=sum_xixj(500,samp_theta)
#Analysis of one [1,2]-element of theta
n_sample=30
mean_theta=np.zeros((1,d**2))
for t in range(n_sample):
    samp_theta=np.ones((d,d),dtype=np.float)
    samp_theta=gen_theta(50,samp_theta,dl_sample)
    samp_xixj=sum_xixj(100,samp_theta)
    samp_theta_vec=np.reshape(samp_theta,(1,d**2))#[t[0,0],t[0,1],..,t[0,d-1],t[1,0],..
    if t==0:
        stack_theta=samp_theta_vec
        stack_x1x2=samp_xixj[1][2]
    else:
        stack_theta=np.vstack((stack_theta,samp_theta_vec))
        stack_x1x2=np.vstack((stack_x1x2,samp_xixj[1][2]))
#visualization of naive sample data, 1 to 3
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter3D(np.ravel(stack_theta[:,1]),np.ravel(stack_theta[:,2]),np.ravel(stack_theta[:,3]),c=np.array(stack_x1x2))
plt.show()

H=np.matrix(np.ones((n_sample,n_sample)))
cov_theta=np.matrix(np.dot(stack_theta,stack_theta.T))
theta2=np.matrix(np.diagonal(cov_theta))
theta2=np.matlib.repmat(theta2,n_sample,1)+np.matlib.repmat(theta2.T,1,n_sample)
hh=2.0
K=np.matrix(np.exp(-(theta2-2*cov_theta)/hh))
K=K - H*K - K*H + H*K*H
la,v=linalg.eig(K)
idx=la.argsort()[::-1]
l1,l2,l3=la[idx[0]],la[idx[1]],la[idx[2]]
v1,v2,v3=v[:,idx[0]],v[:,idx[1]],v[:,idx[2]]
Tpca=np.vstack((np.vstack((v1,v2)),v3))
projected=np.dot(np.matrix(Tpca),np.matrix(K))


base1,base2,base3=np.real(projected[0]),np.real(projected[1]),np.real(projected[2])   
#plt.scatter(np.ravel(base1),np.ravel(base2),c=stack_x1x2)
fig=plt.figure(1)
ax=Axes3D(fig)
ax.scatter3D(np.ravel(base1),np.ravel(base2),np.ravel(base3),c=np.real(stack_x1x2))
p=ax.scatter(np.ravel(base1),np.ravel(base2),np.ravel(base3),c=np.real(stack_x1x2))
fig.colorbar(p)
plt.show()
