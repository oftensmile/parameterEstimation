import numpy as np
import matplotlib.pyplot as  plt
import matplotlib.mlab as mlb
np.random.seed(0)
K,sample,iteration=5,100,100# sample=TOTAL number of sample
alpha_true=np.random.uniform(0,1,K)
alpha_true=alpha_true/np.sum(alpha_true)
xi_true=np.arange(1,K+1)*1.0/K  #1.0/xi_ture is 'scale parameter'
numOfk=np.zeros(K)
for k in range(0,K):
    numOfk[k]=int(sample*alpha_true[k])
    if(k==0):x=np.random.exponential(1.0/xi_true[k],numOfk[k])
    else:x=np.append(x,np.random.exponential(1.0/xi_true[k],size=numOfk))
alpha=np.ones(K)*1.0/K
xi=np.ones(K)*1.0/K
p=np.zeros((len(x),K),dtype=np.float)
sample_sum_of=np.zeros(K,dtype=np.float)
for t in range(iteration):
    #E-step
    for k in range(K):
        p[:,k]=alpha[k]*xi[k]*np.exp(-xi[k]*x)
    for n in range(sample):
        p[n,:]=p[n,:]/np.sum(p[n,:])
    #M-step
    for k in range(K):
        sample_sum_of[k]=np.sum(p[:,k])
        xi[k]=sample_sum_of[k]/np.dot(p[:,k],x) 
    total_sample=sum(sample_sum_of)*1.0
    for k in range(K):
        alpha[k]=sample_sum_of[k]/total_sample
#reusult
print('#alpha_true=',alpha_true)
print('#alpah_est =',alpha)
print('#xi_ture',xi_true)
print('#xi',xi)

num_bin=10
hist,bins=np.histogram(x,bins=num_bin)
width=0.7*(bins[1]-bins[0])
center=(bins[:-1]+bins[1:])/2
plt.bar(center,hist/(2*sample/num_bin),align='center',width=width)
plt.show()

