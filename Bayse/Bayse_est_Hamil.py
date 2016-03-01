import numpy as np
from scipy import linalg 
import matplotlib.pyplot as plt
import copy
np.random.seed(0)
d,T,N,heatN,J,alpha=8,100,1000,20,1,5.0

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
    y=gen_mcmc(100,y,theta)
    for n in range(n_sample):
        y=gen_mcmc(3,y,theta)
        xixj=xixj+np.tensordot(y,y,axes=([0][0]))/n_sample
   #this diagonal condition doesn't required, cause this came from diagonal of theta is zero.
   # for l in range(d):xixj[l][l]=0
    return copy.copy(xixj)

def heat_exixj(indexi,indexj,t_wait,sample,delta_theta,x=[],theta=[[]]):
    sum_expxixj=0
    for t in range(sample):
        x=gen_mcmc(t_wait,x,theta)
        sum_expxixj+=np.exp(-x[indexi]*x[indexj]*delta_theta)
    sum_expxixj/=sample
    return (sum_expxixj, copy.copy(x))

def ratio_p_of_theta(indexi,indexj,sum_of_xixj,delta_ij,theta,valu_heat_exixj):
    a=(1.0/valu_heat_exixj)**delta_ij
    #a=1
    b=sum_of_xixj+alpha*(theta[indexi][indexj]+0.5*delta_ij)
    ratio= a * np.exp(-delta_ij*b)
    return ratio
#################### MAIN #########################
dl_sample=sum_xixj(N,theta)
theta_est=0.1*np.random.rand(d,d)
theta_est=0.5*(theta_est + theta_est.T)

x=np.array(np.ones(d))
idling,waiting,sample_t,sample_x,count=10,3,200,10,1
theta_mean=np.zeros((d,d))
vec_corelation=np.reshape(copy.copy(dl_sample),(1,d**2))

for t in range(idling+waiting*sample_t):
    for i in range(d):
        for j in range(i,d):
            delta_theta=np.random.randn()#proposal
            theta_est_vec_pre=np.reshape(copy.copy(theta_est),(1,d**2))
            theta_est[i][j]+=delta_theta
            theta_est[j][i]+=delta_theta
            theta_est_vec=np.reshape(copy.copy(theta_est),(1,d**2))
            
            A=heat_exixj(i,j,waiting,sample_x,delta_theta,x,theta_est) 
            
            a=A[0]**N
            x=A[1]
            r=np.dot(vec_corelation,(theta_est_vec-theta_est_vec_pre).T)+0.5*alpha*(np.dot(theta_est_vec,theta_est_vec.T)-np.dot(theta_est_vec_pre,theta_est_vec_pre.T)-2*np.sum(theta_est_vec-theta_est_vec_pre))
            r=np.exp(-r) / a

            R=np.random.uniform(0,1)
            
            if(R>r):
                theta_est[i][j]-=delta_theta
                theta_est[j][i]-=delta_theta

    if(idling<=t and t%waiting==0 ):
        theta_mean=theta_mean+theta_est
        count+=1
    """
    if(t==50):
        theta50=copy.copy(theta_est)
    elif(t==100):
        theta100=copy.copy(theta_est)
    elif(t==10):
        theta150=copy.copy(theta_est)
    """
    a=np.sum(abs(theta_mean/count- theta))
    print(a,"#residual of theta",t)

theta_mean=theta_mean/count

print("theta_est=",theta_est)
plt.subplot(221)
plt.imshow(theta)
plt.colorbar()
plt.title('true theta')
plt.subplot(222)
plt.imshow(dl_sample)
plt.colorbar()
plt.title('corelation of spins(Gram mat)')
plt.subplot(223)
plt.imshow(theta_est)
plt.colorbar()
plt.title('final ')
plt.subplot(224)
plt.imshow(theta_mean)
plt.colorbar()
plt.title('Bayse estimation(mean of theta est)')
plt.show()
