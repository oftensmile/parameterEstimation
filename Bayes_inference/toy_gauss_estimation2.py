import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt                                                                                                 
np.random.seed(0)                                                                                                               
                                                                                                                                   
   #this is estimation of gauss pdf                                                                                                
N,mu0,sig0=1000,2.3,1.0                                                                                                         
x=mu0+sig0*np.random.randn(N)                                                                                                                                                                                                                                      
T=500                                                                                                                           
mu,sig=0.5,sig0                                                                                                              
mu_pri,sig_pri=0,2,    #unknown valuable is only mu                                                                                                          
sample_sum=np.sum(x)          
waiting,sample=3,1000
mu_mean,count=0,0
for t in range(T+waiting*sample):                                                                                                              
    rand_mu,rand_th=np.random.randn(1),np.random.randn(1)**2                                                                    
    R_mu,R_th=np.random.uniform(0,1),np.random.uniform(0,1)                                                                                                                                                                                                  
    valu=0.5*(1.0/sig**2)*N*(2*mu*rand_mu+rand_mu**2)-(1.0/sig**2)*sample_sum*rand_mu+0.5*(2*mu*rand_mu+ rand_mu**2-2*rand_mu*mu_pri)               
    r=np.exp(-valu)                                                                                                             
    if(R_mu<=r):                                                                                                                
        mu+=rand_mu                                                                                                             
    if(T<t and t%waiting==0):
        mu_mean+=mu
        count+=1
        #print(count,mu_mean/count)
mu_mean/=count
print(mu0,"#mu_true",mu0)
print(mu_mean[0],"#mu_beyes")
print(sample_sum/N,"#mu_mle")
