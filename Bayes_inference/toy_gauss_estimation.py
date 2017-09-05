import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt                                                                                                 
np.random.seed(0)                                                                                                               
                                                                                                                                   
   #this is estimation of gauss pdf                                                                                                
N,mu0,sig0=1000,1.3,1.8                                                                                                         
x=mu0+sig0*np.random.randn(N)                                                                                                                                                                                                                                      
T=500                                                                                                                           
mu,sig=0.5,1.0                                                                                                                  
mu_pri,sig_pri=0,2                                                                                                              
th,th_pri=1.0/sig**2,1.0/sig_pri**2                                                                                             
sample_sum=np.sum(x)                                                                                                            
for t in range(T):                                                                                                              
    rand_mu,rand_th=np.random.randn(1),np.random.randn(1)**2                                                                    
    R_mu,R_th=np.random.uniform(0,1),np.random.uniform(0,1)                                                                                                                                                                                                  
    valu=0.5*th*N*(2*mu*rand_mu+rand_mu**2)-th*sample_sum*rand_mu+0.5*(2*mu*rand_mu+ rand_mu**2-2*rand_mu*mu_pri)               
    r=np.exp(-valu)                                                                                                             
    if(R_mu<=r):                                                                                                                
        mu+=rand_mu                                                                                                             
    valu=0.5*((sample_sum-N*mu)*rand_th + (rand_th**2+2*th*rand_th-2*th_pri*rand_th))                                           
    r=np.exp(-valu)                                                                                                             
  if(R_th<=r):                                                                                                                
        th                                                                                                                      

