#optional method is that if difference between true and est in each class is to be smaller than epc2, then stop to update only for the parameter of the class
import numpy as np
import matplotlib.pyplot as  plt
import matplotlib.mlab as mlb
from scipy.stats import expon
import copy
np.random.seed(0)
K,sample,iteration=3,1000,100# sample=TOTAL number of sample
alpha_true=np.random.uniform(0,1,K)
alpha_true=alpha_true/np.sum(alpha_true)
beta_true=2*np.arange(1,K+1)*1.0/K  #1.0/xi_ture is so called 'scale parameter'
#alpha_est=np.ones(K)*1.0/K
#xi_est=np.ones(K)*1.0/K
numOfk=np.zeros(K)
alpha_est=np.random.uniform(0,1,K)
beta_est=np.random.uniform(0,1,K)
for k in range(K):
    numOfk[k]=int(sample*alpha_true[k])
    if(k==0):x=np.random.exponential(beta_true[k],numOfk[k])
    else:x=np.append(x,np.random.exponential(beta_true[k],numOfk[k]))
total_sample=int(len(x))
p=np.zeros((total_sample,K),dtype=np.float)
ganma=np.zeros((total_sample,K),dtype=np.float)
sample_sum_of=np.zeros(K,dtype=np.float)
epc=0.00001# precision-parameter
for t in range(iteration):
    #E-step
    for k in range(K):
        p[:,k]=alpha_est[k]/beta_est[k]*np.exp(-x/beta_est[k])
    for n in range(int(total_sample)):
        sum_p_of_n=np.sum(p[n,:])
        if(epc<=sum_p_of_n):ganma[n,:]=copy.copy(p[n,:])/sum_p_of_n
        else: ganma[n,:]=np.zeros((1,K),dtype=np.float)
    #M-step
    for k in range(K):
        sample_sum_of[k]=np.sum(copy.copy(ganma[:,k]))
        beta_est[k]=np.dot(ganma[:,k],x) / sample_sum_of[k]
        print(np.abs(beta_true[k]- beta_est[k]))
    total_sample=sum(sample_sum_of)
    for k in range(K):
        alpha_est[k]=sample_sum_of[k]/total_sample
    #print(np.sum(np.abs(alpha_true-alpha_est)),'',np.sum(np.abs(beta_true - beta_est)))
    print(' '.join(map(str,np.abs(alpha_true-alpha_est))),' '.join(map(str,np.abs(beta_true - beta_est))))
    

#reusult
print('#alpha_true=',alpha_true)
print('#alpah_est =',alpha_est)
print('#beta_ture',beta_true)
print('#beta',beta_est)

fig,ax=plt.subplots(1,1)
mean, var, skew ,kurt=expon.stats(moments='mvsk')
X=np.linspace(min(x),max(x),)
for k in range(K):
    ax.plot(X,0.5*sample*expon.pdf(X,loc=0,scale=beta_true[k]),'r--',lw=2,label='frozen pdf')
    ax.plot(X,0.5*sample*expon.pdf(X,loc=0,scale=beta_est[k]),'b--',lw=2,label='frozen pdf')
num_bin=10
hist,bins=np.histogram(x,bins=num_bin)
width=0.6*(bins[1]-bins[0])
center=(bins[:-1]+bins[1:])/2
ax.hist(x,bins,range=(min(x),max(x)),facecolor='b',alpha=0.4)
plt.show()


