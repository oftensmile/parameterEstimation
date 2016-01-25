import numpy as np
import matplotlib.pyplot as plt
sqrt = np.emath.sqrt
from scipy.optimize import fsolve
from scipy.optimize import root
np.random.seed(0)
d=10
def mcmc_sampling_given_J(mt,J,x=[]):
    l=len(x)
    for t in range(mt):
        for i in range(l):
            valu=(( x[(l+i-1)%l]+x[(i+1)%l] ))
            r=np.exp(-J*valu)/(np.exp(J*valu)+np.exp(-J*valu))
            R=np.random.uniform(0,1)
            if(R<=r):
                x[i]=1
            else:
                x[i]=-1
    return x

def regres_from_mcmc(k,nJ,Jmax,ns):#nJ=sample num
    dJ=Jmax/nJ
    J,PhiofJ=np.zeros(nJ),np.zeros(nJ)
    for j in range(nJ):
        x0=np.ones(d)
        x0=mcmc_sampling_given_J(1000,dJ*j,x0)
        m=0
        for t in range(ns):
            x0=mcmc_sampling_given_J(3,dJ*j,x0)
            if(t==0):
                x=x0
            else:
                x=np.vstack((x,x0))
        m=calcPhi(x)
        J[j],PhiofJ[j]=dJ*j,m
    z=np.polyfit(J,PhiofJ,k)
    return z

def phi_eq(J,m,z=[]):
    p=np.poly1d(z)  # this may cannot use for root() function
    return p(J)-m

def mcmc_sampling_given_h(mt,h,x=[],tu=[]):
    l=len(x)
    for t in range(mt):
        for i in range(l):
            r=np.exp(-h*tau[i])/(np.exp(h*tau[i])+np.exp(-h*tau[i]))
            R=np.random.uniform(0,1)
            if(R<=r):
                x[i]=1
            else:
                x[i]=-1
    return x

def calcPhi(x=[[]]):
    m,ns=0, len(x[0])
    for n in range(ns):
        value=0
        for i in range(d):
            value+=x[n][i]*x[n][(i+1)%d]
        m+=value
    return m/ns

def calc_xtau(tau=[],x=[[]]):
    s,ns=0,len(x[0])
    for n in range(ns):
       s+=np.dot(tau,x[n])/d 
    return s/ns

##############main###############
k,nJ,Jmax,ns=30,100,10,300
Q=100
tau=[-1 if i%3==0 else 1 for i in range(d)]#[-1,1,1-1,,...,1]
# find Phi of J
z=regres_from_mcmc(k,nJ,Jmax,ns)
J,h=0.2,0.1
for q in range(Q):
    x0=np.ones(d)
    x0=mcmc_sampling_given_h(1000,h,x0,tau)
    for l in range(ns):
        x0=mcmc_sampling_given_h(3,h,x0,tau)
        if (l==0):
            x=x0
        else:
            x=np.vstack((x,x0))
    #E-step
    m=calcPhi(x)
    s=calc_xtau(tau,x)
    #M-step
    h=np.arctanh(s)
    J=root(phi_eq,0.1,(m,z))
    J0=np.asscalar(J.x)
    print(q," ", s," ",h," ",J0)
