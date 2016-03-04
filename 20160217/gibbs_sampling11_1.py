import numpy as np
from scipy import linalg 
import matplotlib.pyplot as plt
import copy
np.random.seed(0)
d,T,N,heatN,J,alpha=8,100,1000,20,1,40.0

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
    return copy.copy(x)

def sum_xixj(n_sample,theta=[[]]):
    xixj=np.zeros((d,d))
    y=np.ones(d)
    y=gen_mcmc(500,y,theta)
    for n in range(n_sample):
        y=gen_mcmc(3,y,theta)
        xixj=xixj+np.tensordot(y,y,axes=([0][0]))/n_sample
   #this diagonal condition doesn't required, cause this came from diagonal of theta is zero.
   # for l in range(d):xixj[l][l]=0
    return copy.copy(xixj)

def heat_exixj(indexi,indexj,t_wait,sample,x=[],theta=[[]]):
    sum_expxixj=0
    for t in range(sample):
        x=gen_mcmc(t_wait,x,theta)
        sum_expxixj+=np.exp(-x[indexi]*x[indexj])
    sum_expxixj/=sample
    return (sum_expxixj, copy.copy(x))

def ratio_p_of_theta(indexi,indexj,sum_of_xixj,delta_ij,theta,valu_heat_exixj):
    vec_theta=np.array(np.reshape(theta,(1,d**2)))
    a=(1.0/valu_heat_exixj)**delta_ij
    b=sum_of_xixj+alpha*(theta[indexi][indexj]+0.5*delta_ij)
    ratio= a * np.exp(-delta_ij*b)
    return ratio
#################### MAIN #########################
dl_sample=sum_xixj(N,theta)
# initial theta
theta_est=0.1*np.random.rand(d,d)
theta_est=0.5*(theta_est + theta_est.T)

x=np.array(np.ones(d))
sample_x=20
idling,waiting,sample_t,count=10,3,300,0
theta_mean=np.zeros((d,d))

"""
check of the corelation of theta
"""
correlation=0.0
for t in range(idling+waiting*sample_t):
    #pre_tehta=copy.copy(theta_mean)
    for i in range(d):
        for j in range(i,d):
            delta_theta=np.random.randn()#proposal
            if t==0:
                ret=heat_exixj(i,j,100,sample_x,x,theta_est)
                valu_heat_exixj=ret[0]
                x=ret[1]
            else:
                ret=heat_exixj(i,j,3,sample_x,x,theta_est)
                valu_heat_exixj=ret[0]
                x=ret[1]
            r=ratio_p_of_theta(i,j,dl_sample[i][j],delta_theta,theta_est,valu_heat_exixj)
            R=np.random.uniform(0,1)
            if(R<=r):
                theta_est[i][j]+=delta_theta
                theta_est[j][i]+=delta_theta
            
    if(idling<=t and t%waiting==0 ):
        theta_mean=theta_mean+theta_est
        count+=1
        #print(np.sum(np.absolute(theta_mean/count-theta)))
    #a=(copy.copy(pre_tehta)).reshape((1,d**2))*(copy.copy(theta_mean)).reshape((1,d**2))
    #correlation=np.sum(a)/d**2
    a=np.sum(abs(theta_mean))
    print(a,"#sum of absolute theta_mean",t)

theta_mean=theta_mean/count



"""loss,delta=np.zeros(T),np.zeros(T)
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
result=[gen_mcmc(1000,np.ones(d),theta_est)]
"""
print("theta_est=",theta_est)
plt.subplot(221)
plt.imshow(theta)
plt.colorbar()
plt.title('true theta')
plt.subplot(222)
plt.imshow(theta_est)
plt.colorbar()
plt.title('estimated theta ')
plt.subplot(223)
plt.imshow(dl_sample)
plt.colorbar()
plt.title('corelation of spins')
plt.show()
