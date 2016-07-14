import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
np.random.seed(0)

alpha,beta = 1.0 , 2.0
b=0.9
vari_porp=0.1
dim=2
interval=30
tot_sample,removed=100,0

def accept(x=[]):
    return (1-alpha)*np.exp(-beta*((x[0]+1)**2 *b**2 + (x[1]+1)**2))+alpha*np.exp(-beta*((x[0]-1)**2 *b**2 + (x[1]-1)**2))

def simple_mcmc(x=[]):
    propose=np.copy(x)+vari_porp*np.random.randn(dim)
    accept_ratio=accept(propose)/accept(x)#np.exp(-beta*(energy_func(propose)-energy_func(x)))
    # Metropolice Hesting method
    u=np.random.uniform()
    if(accept_ratio>1):
        x=np.copy(propose)
    elif(u<accept_ratio):
        x=np.copy(propose)
    return x

#if __name__ =='__main__':
x=[-10.0,10.0]
for i in range(tot_sample+removed):
    for t in range(interval):
        x=simple_mcmc(x)
    if(i==removed):
        X_tot=np.copy(x)
    elif(i>removed):
        print(i,x[0],x[1])
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
fig.savefig("scatter.png")
fig.show()
