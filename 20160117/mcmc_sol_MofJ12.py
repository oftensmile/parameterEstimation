import numpy as np
import matplotlib.pyplot as plt
sqrt = np.emath.sqrt
from scipy.optimize import fsolve
from scipy.optimize import root
np.random.seed(0)
d=32#(#total=d*d)
def mcmc_sampling_given_J(mt,J,x=[[]]):
    l=len(x)
    for t in range(mt):
        for i1 in range(l):
            for i2 in range(l):
                valu=( x[(l+i1-1)%l][i2]+x[(i1+1)%l][i2]+x[i1][(l+i2-1)%l]+x[i1][(i2+1)%l] )
                r=np.exp(-J*valu)/(np.exp(J*valu)+np.exp(-J*valu))
                R=np.random.uniform(0,1)
                if(R<=r):
                    x[i1][i2]=1
                else:
                    x[i1][i2]=-1
    return x

def regres_from_mcmc(k,nJ,Jmax,ns):#nJ=sample num
    dJ=Jmax/nJ
    J,PhiofJ=np.zeros(nJ),np.zeros(nJ)
    for j in range(nJ):
        x0=np.ones((d,d))
        x0=mcmc_sampling_given_J(500,dJ*j,x0)
        m=0
        for t in range(ns):
            x0=mcmc_sampling_given_J(3,dJ*j,x0)
            if(t==0):
                x=x0
            else:
                x=np.vstack((x,x0))# [[Matrix],[Matrix],....,[Matrix]].T
        m=calcPhi(x)
        J[j],PhiofJ[j]=dJ*j,m
    z=np.polyfit(J,PhiofJ,k)
    #p=np.poly1d(z)
    #Jpoint=np.array(J)
    #Jp=np.linspace(0,Jmax,100)
    #plt.plot(Jpoint,p(Jpoint),'*',Jp,p(Jp),'-')
    #plt.ylim(-20,0)
    #plt.show()
    return z

def phi_eq(J,m,z=[]):
    p=np.poly1d(z)  # this may cannot use for root() function
    return p(J)-m

def mcmc_sampling_given_h(mt,h,x=[[]],tu=[[]]):
    l,ns=len(x.T),int(len(x)/d)
    for t in range(mt):
        for i1 in range(ns*l):
            for i2 in range(l):
                r=np.exp(-h*tau[i1][i2])/(np.exp(h*tau[i1][i2])+np.exp(-h*tau[i1][i2]))
                R=np.random.uniform(0,1)
                if(R<=r):
                    x[i1][i2]=1
                else:
                    x[i1][i2]=-1
    return x
def mcmc_sampling_given_hJ(mt,h,J,x=[[]],tu=[[]]):
    l,ns=len(x.T),int(len(x)/d)
    for t in range(mt):
        for i1 in range(l*ns):
            for i2 in range(l): 
                valu = J*( x[(l+i1-1)%l][i2]+x[(i1+1)%l][i2]+x[i1][(i2+l-1)%l]+x[i1][(i2+1)%l] ) +h*tau[i1][i2]
                r=np.exp(-valu)/(np.exp(valu)+ np.exp(-valu))
                R=np.random.uniform(0,1)
                if(R<=r):
                    x[i1][i2]=1
                else:
                    x[i1][i2]=-1
    return x    

def calcPhi(x=[[]]):
    m,ns=0, len(x)//len(x.T)
    for n in range(ns):
        m=0
        for i1 in range(d):
            for i2 in range(d):
                m+=x[n*d+i1][i2]*x[n*d+i1][(i2+1)%d]+x[n*d+i1][i2]*x[n*d+(i1+1)%d][i2]
    return m/ns

def calc_xtau(tau=[[]],x=[[]]):
    s,ns=0,int(len(x)/d)
    for n in range(ns):
        s=0
        for i1 in range(d): 
            s+=np.dot(tau[i1],x[n*d+i1])/(d*d) 
    return s/ns

##############main###############
k,nJ,Jmax,ns=30,100,10,300
Q=50
Jtrue,htrue=1.1,0.3

#test=[[-1*(-1)**j if i%3==0 else 1*(-1)**j for i in range(d)] for j in range(d)]#[-1,1,1-1,,...,1],[same]*(-1),[same]*(-1)^2,...
#Original Image
original=np.ones((d,d))
original=mcmc_sampling_given_J(500,Jtrue,original)
#Dameged Process
tau=np.ones((d,d))
tau=mcmc_sampling_given_h(500,htrue,tau,original)
damaged=tau

z=regres_from_mcmc(k,nJ,Jmax,ns)

J,h=0.2,0.1
for q in range(Q):
    x0=np.ones((d,d))
    x0=mcmc_sampling_given_hJ(500,h,J,x0,tau)
    for l in range(ns):
        x0=mcmc_sampling_given_hJ(3,h,J,x0,tau)
        if (l==0):
            x=x0
        else:
            x=np.vstack((x,x0))
    #E-step
    m=calcPhi(x)
    s=calc_xtau(tau,x)
    #M-step
    h=np.arctanh(s)
    tempJ=root(phi_eq,0.1,(m,z))
    J=np.asscalar(tempJ.x)
    print(q," ",h," ",J)

latent=x[len(x)-d:len(x)][0:d]
#Image Restortion
restored=np.ones((d,d))
restored=mcmc_sampling_given_J(200,J,restored)

plt.subplot(241)
plt.title("original")
plt.imshow(np.array(original),interpolation='none')
plt.subplot(242)
plt.title("damaged")
plt.imshow(damaged,interpolation='none')
plt.subplot(243)
plt.title("latent")
plt.imshow(latent,interpolation='none')
plt.subplot(244)
plt.title("restored")
plt.imshow(np.array(restored),interpolation='none')
plt.savefig("test.png")


#plt damaged
#p=np.poly1d(z)
#Jp=np.linspace(0,Jmax,100)
#plt.plot(Jp,p(Jp),'-',Jp,p(J)*np.zeros(100),'--')
#plt.ylim(-20,1)
#plt.show()

