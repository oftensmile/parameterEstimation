import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
np.random.seed(0)

beta=1.0
bariance=0.1
t_burnin=20
epc_max=3000

def accept(dim,J,x=[]):
    function=0
    for i in range(dim):
        function+=(1.0/dim)*(x[i]-J[i])**2
    return np.exp(-beta*function)

def simple_mcmc(dim,x=[]):
    J=np.ones(dim)
    propose=np.copy(x)+x*np.random.choice([-2,0],dim)#bariance*np.random.randn(dim)
    accept_ratio=accept(dim,J,propose)/accept(dim,J,x)#np.exp(-beta*(energy_func(propose)-energy_func(x)))
    # Metropolice Hesting method
    u=np.random.uniform()
    if(accept_ratio>1.0):
        x=np.copy(propose)
    elif(u<accept_ratio):
        x=np.copy(propose)
    return x  

def mpf(name,dim,n_sample,n_remove,t=[],x_tot=[[]]):
    J=np.ones(dim)  #It is prepaiered of the check
    len_sample=len(x_tot)
    error=100
    f=open(name,'w')
    for epc in range(epc_max):
        prob_flow=np.zeros(dim)
        for n in range(len_sample):
            x_n=np.copy(x_tot[n])
            accept_of_x_n=accept(dim,t,x_n)
            for d in range(dim):
                x_n_d=np.copy(x_n)
                x_n_d[d]*=-1
                prob_flow=prob_flow-((t-x_n_d)-(t-x_n))*(accept(dim,t,x_n_d)/accept_of_x_n)**(0.5)/dim
        prob_flow/=len_sample
        t=t-prob_flow
        error_pre=error
        error=np.sum(np.abs(t-J))/dim
        f.write(str(error)+"\n")
        if(error_pre<error):
            f.close()
            break    
    return t
if __name__ == '__main__':
    n_remove=100
    for n_sample in [10,30,50,100,200]:
        for dim in [4,8,16]:
            name="sample"+str(n_sample)+"-dim"+str(dim)+".dat"
            ##   SAMPLING PROCESS    ##
            x=np.random.choice([-1,1],dim)
            for s in range(n_sample+n_remove):
                for t in range(t_burnin):
                    x=simple_mcmc(dim,x)
                if(s==n_remove):
                    x_tot=x
                elif(s>n_remove):
                    x_tot=np.vstack((x_tot,np.copy(x)))
            ##      MPF     ##
            J_mpf=np.random.rand(dim)
            J_mpf=mpf(name,dim,n_sample,n_remove,np.copy(J_mpf),x_tot)
