import numpy as np
import matplotlib.pyplot as  plt
import matplotlib.mlab as mlb
np.random.seed(0)
K,sample,iteration=3,1000,500# sample=TOTAL number of sample
alpha_true=np.random.uniform(0,1,K)
alpha_true=alpha_true/np.sum(alpha_true)
beta_true=np.arange(1,K+1)*1.0/K  #1.0/xi_ture is so called 'scale parameter'
#alpha_est=np.ones(K)*1.0/K
#xi_est=np.ones(K)*1.0/K
numOfk=np.zeros(K)
alpha_est=np.random.uniform(0,1,K)*1.0/K
beta_est=np.random.uniform(0,1,K)*1.0/K
for k in range(K):
    numOfk[k]=int(sample*alpha_true[k])
    if(k==0):x=np.random.exponential(beta_true[k],numOfk[k])
    else:x=np.append(x,np.random.exponential(beta_true[k],size=numOfk[k]))
total_sample=int(len(x))
p=np.zeros((total_sample,K),dtype=np.float)
ganma=np.zeros((total_sample,K),dtype=np.float)
sample_sum_of=np.zeros(K,dtype=np.float)
for t in range(iteration):
    #E-step
    for k in range(K):
        p[:,k]=alpha_est[k]/beta_est[k]*np.exp(-x/beta_est[k])
    for n in range(int(total_sample)):
        ganma[n,:]=p[n,:]/np.sum(p[n,:])
    #M-step
    for k in range(K):
        sample_sum_of[k]=np.sum(ganma[:,k])
        beta_est[k]=np.dot(ganma[:,k],x) / sample_sum_of[k]
    total_sample=sum(sample_sum_of)
    for k in range(K):
        alpha_est[k]=sample_sum_of[k]/total_sample
#reusult
print('#alpha_true=',alpha_true)
print('#alpah_est =',alpha_est)
print('#beta_ture',beta_true)
print('#beta',beta_est)

num_bin=10
hist,bins=np.histogram(x,bins=num_bin)
width=0.7*(bins[1]-bins[0])
center=(bins[:-1]+bins[1:])/2
plt.bar(center,hist/(2*sample/num_bin),align='center',width=width)
plt.show()

