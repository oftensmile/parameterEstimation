import numpy as np
from scipy import linalg 
import matplotlib.pyplot as plt
np.random.seed(0)
d,T,N,heatN,J=8,0,1000,20,1
theta=[[1 if i==(j+1+d)%d or i==(j-1+d)%d else 0 for i in range(d)] for j in range(d)]
alpha=5.0
prior_A=alpha*np.matrix(np.eye(d**2))
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
    y=gen_mcmc(100,y,theta)
    for n in range(n_sample):
        y=gen_mcmc(3,y,theta)
        xixj=xixj+np.tensordot(y,y,axes=([0][0]))/n_sample
    for l in range(d):xixj[l][l]=0
    return xixj

def heat_exixj(id_i,id_j,ideling,t_wait,sample,x=[],theta=[[]]):
    sum_expxixj=0
    x_temp=x
    x_temp=gen_mcmc(ideling,x_temp,theta)
    for t in range(sample):
        x_temp=gen_mcmc(t_wait,x_temp,theta)
        sum_expxixj+=np.exp(-x_temp[id_i]*x_temp[id_j])
    sum_expxixj/=sample
    return sum_expxixj
#################### MAIN #########################
dl_sample=sum_xixj(N,theta)
theta_est=0.1*np.random.rand(d,d)
theta_est=0.5*(theta_est + theta_est.T)
ideling,waiting,sample_t,heat_sample,count=100,3,300,40,0
theta_mean=np.zeros((d,d))
for t in range(ideling+waiting*sample_t):
    for i in range(d):
        for j in range(i,d):
            delta_theta=np.random.randn(1)#proposal
            a=heat_exixj(i,j,ideling,waiting,heat_sample,np.ones(d),theta_est)
            a=a**(-delta_theta)
            b=dl_sample[i][j]+0.5*alpha*( 2*theta_est[i][j]+delta_theta)
            b=np.exp(-delta_theta*b)
            r=a*b
            R=np.random.uniform(0,1)
            if(R<=r):
                theta_est[i][j]+=delta_theta
                theta_est[j][i]+=delta_theta
    if(ideling<=t and t%waiting==0 ):
        theta_mean=theta_mean+theta_est
        count+=1
        print(np.sum(np.absolute(theta_mean)/count))
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
#print("theta_est=",theta_est)
plt.subplot(221)
plt.imshow(theta)
plt.colorbar()
plt.title('true theta')
plt.subplot(222)
plt.imshow(theta_mean)
plt.colorbar()
plt.title('estimated theta ')
plt.show()
