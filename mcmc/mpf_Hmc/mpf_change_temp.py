import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
np.random.seed(0)

dim=6
beta=1.0
bariance=0.1
n_sample,n_remove=100,100
t_burnin=20
epc_max=3000

def accept(J,x=[]):
    function=0
    for i in range(dim):
        function+=(1.0/dim)*(x[i]-J[i])**2
    return np.exp(-beta*function)

def simple_mcmc(x=[]):
    J=np.ones(dim)
    propose=np.copy(x)+x*np.random.choice([-2,0],dim)#bariance*np.random.randn(dim)
    accept_ratio=accept(J,propose)/accept(J,x)#np.exp(-beta*(energy_func(propose)-energy_func(x)))
    # Metropolice Hesting method
    u=np.random.uniform()
    if(accept_ratio>1.0):
        x=np.copy(propose)
    elif(u<accept_ratio):
        x=np.copy(propose)
    return x  

def mpf(t=[],x_tot=[[]]):
    J=np.ones(dim)  #It is prepaiered of the check
    len_sample=len(x_tot)
    error=100
    t=np.random.rand(dim)
    for alpha in [0.1,0.5,1.0]:
        name="temp-scaled"+str(alpha)+".dat"
        f=open(name,"w")
        for epc in range(epc_max):
            prob_flow=np.zeros(dim)
            for n in range(len_sample):
                x_n=np.copy(x_tot[n])
                accept_of_x_n=accept(t,x_n)
                for d1 in range(dim):
                    x_n_d=np.copy(x_n)
                    x_n_d[d1]*=-1
                    for d2 in range(dim):
                        if(d2!=d1):
                            x_n_d[d2]*=-1
                            prob_flow=prob_flow-(-x_n_d+x_n)*(accept(t,x_n_d)/accept_of_x_n)**alpha/(dim**2)
                        #prob_flow=prob_flow-((t-x_n_d)-(t-x_n))*(accept(t,x_n_d)/accept_of_x_n)**(0.5)/dim**2
            prob_flow/=len_sample
            t=t-0.1*prob_flow
            error_pre=error
            error=np.sum(np.abs(t-J))/dim
            f.write(str(error)+"\n")
            #print(error)
            if(error_pre<error):
                f.close()
                break    
    return t
if __name__ == '__main__':
    ##   SAMPLING PROCESS    ##
    x=np.random.choice([-1,1],dim)
    for s in range(n_sample+n_remove):
        for t in range(t_burnin):
            x=simple_mcmc(x)
        if(s==n_remove):
            x_tot=x
        elif(s>n_remove):
            x_tot=np.vstack((x_tot,np.copy(x)))
    ##      MPF     ##
    J_mpf=np.random.rand(dim)
    print("#",J_mpf)
    J_mpf=mpf(np.copy(J_mpf),x_tot)
    print("#",J_mpf)
