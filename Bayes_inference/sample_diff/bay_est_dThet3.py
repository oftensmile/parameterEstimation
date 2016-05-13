import numpy as np
from scipy import linalg 
import matplotlib.pyplot as plt
import copy
np.random.seed(0)
d,T,N,heatN,J=8,50,1000,20,1
theta=[[1 if i==(j+1+d)%d or i==(j-1+d)%d else 0 for i in range(d)] for j in range(d)]
alpha=5.0
prior_A=alpha*np.matrix(np.eye(d**2))
def gen_mcmc(t_wait, x=[],theta=[[]]):
    for t in range(t_wait):
        for i in range(d):
            y=copy.copy(x)
            valu=J*(np.dot(theta[:][i],y)-y[i]*theta[i][i])
            r=np.exp(-valu)
            R=np.random.uniform(0,1)
            if(R<=r):
                x[i]=1
            else:
                x[i]=-1
    return copy.copy(x)

def sum_xixj(n_sample,theta=[[]]):
    xixj=np.zeros((d,d))
    y=np.ones(d)
    y=gen_mcmc(100,y,theta)
    for n in range(n_sample):
        y=gen_mcmc(5,y,theta)
        xixj=copy.copy(xixj)+np.tensordot(y,y,axes=([0][0]))
    for l in range(d):xixj[l][l]=0
    return copy.copy(xixj)/n_sample

def heat_exixj(id_i,id_j,ideling,t_wait,sample,theta=[[]]):
    sum_expxixj=0
    x=np.ones(len(theta[0]))
    x=gen_mcmc(ideling,x,theta)
    for t in range(sample):
        x=gen_mcmc(t_wait,x,theta)
        sum_expxixj+=np.exp(-x[id_i]*x[id_j])
    sum_expxixj/=sample
    return sum_expxixj
#################### MAIN #########################
dl_sample=sum_xixj(N,theta)
theta_est=0.1*np.random.rand(d,d)
theta_est=0.5*(theta_est + theta_est.T)
ideling,waiting,sample_t,heat_sample,count=100,10,300,30,0
theta_mean=np.zeros((d,d))
#f=open('loss.dat','w')
for t in range(ideling+waiting*sample_t):
    for i in range(d):
        for j in range(i,d):
            delta_theta=np.random.randn(1)#proposal
            a=heat_exixj(i,j,ideling,waiting,heat_sample,theta_est)
            a=a**(-delta_theta)
            b=dl_sample[i][j]+alpha*( theta_est[i][j]+0.5*delta_theta)
            r= a * np.exp(-delta_theta*b)
            R=np.random.uniform(0,1)
            if(R<=r):
                theta_est[i][j]+=delta_theta
                theta_est[j][i]+=delta_theta
    if(ideling<=t and t%waiting==0 ):
        theta_mean=theta_mean+theta_est
        count+=1
        print(np.sum(np.absolute(theta_mean/count-theta_est)))
#f.write(str(np.sum(np.absolute(theta_mean)/count)))
theta_mean=theta_mean/count
#loss,delta=np.zeros(T),np.zeros(T)
#for l in range(d):theta_est[l][l]=0
#using AdaGrad
#r,a,epc=0,1,0.001
#for k in range(T):
#    dl_model=sum_xixj(heatN,theta_est)
    #r+=np.sum(dl_model*dl_model)
    #lr=a/(np.sqrt(r)+epc)
#    lr=1.0/np.log(2+k)
#    grad=dl_sample - dl_model
#    theta_est=theta_est-lr*grad
#    loss[k]=np.absolute(theta-theta_est).sum()
#    delta[k]=np.absolute(grad).sum()

result=[gen_mcmc(1000,np.ones(d),theta_est)]
plt.subplot(221)
plt.imshow(theta)
plt.colorbar()
plt.title('true theta')
plt.subplot(222)
plt.imshow(theta_mean)
plt.colorbar()
plt.title('bayse estimation')
plt.subplot(223)
plt.imshow(dl_sample)
plt.colorbar()
plt.title('corelation of spin')
plt.show()
