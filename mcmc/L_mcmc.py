import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
np.random.seed(0)

beta = 2.0
b=0.9
h=0.1
dim=2
rL=0.1
interval=30
tot_sample,removed=1000,0

def accept(x=[],p=[]):
    return np.exp(-beta*((x[0]-1)**2 *b**2 + (x[1]-1)**2 +
        p[0]**2+p[1]**2
        ))

def H_mcmc(x=[],p=[]):
    ##      Using Leap-Frog Method     ##
    p=np.random.randn(dim)
    p_propose=p-0.5*h*np.array([b**2*(x[0]-1),x[1]-1])
    #       Applied Lnagevin Dynamics
    x_propose=x+h*np.copy(p_propose)+rL*np.random.randn(dim)
    p_propose=p_propose-0.5*h*np.array([b**2*(x_propose[0]-1),x_propose[1]-1])
    accept_ratio=accept(x_propose,p_propose)/accept(x,p)#np.exp(-beta*(energy_func(propose)-energy_func(x)))
    # Metropolice Hesting method
    u=np.random.uniform()
    if(accept_ratio>1):
        x=np.copy(x_propose)
        p=np.copy(p_propose)
    elif(u<accept_ratio):
        x=np.copy(x_propose)
        p=np.copy(p_propose)
    return [x,p]

#if __name__ =='__main__':
x=[-10.0,-10.0]
p=[1.0,1.0]
for i in range(tot_sample+removed):
    for t in range(interval):
        x,p=H_mcmc(x,p)
    if(i==removed):
        X_tot=np.copy(x)
    elif(i>removed):
        print(i-removed,x[0],x[1])
        X_tot=np.vstack((X_tot,np.copy(x)))
#Make the data to visualize
x1=np.copy(X_tot.T[0])
x2=np.copy(X_tot.T[1])
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x1,x2)
ax.set_title('scatter plot')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
fig.savefig("scatter_Langevin.png")
fig.show()
