import numpy as np
import matplotlib.pyplot as  plt
#from scipy.optimize import fsolve
sqrt = np.emath.sqrt
from scipy.optimize import fsolve
from scipy.optimize import root
np.random.seed(0)
d=10

def func(x):
    return x*np.cos(x-4)
x0=fsolve(func,0.0)
x0=fsolve(func, -0.74)
# calc partition func

def f(x):
    d=2
    return x**2-d*x+1
def g(x,d=[]):
    return x**2-d[0]*d[1]*x+1
# decimal to bnary
def deciToBy(x,d):
    s='#0'+str(d)+'b'
    num=format(x,s)
    a=num.split('b')
    a=a[1]
    v=[int(a[i:i+1]) for i in range(len(a))]
    v=np.array(v)
    v=2*(v-0.5*np.ones(len(v)))
    return v

def f(J,x=[]):
    theta = np.zeros((d,d),dtype=np.float)
    for i in range(d):
        theta[i][(i+1)%d]=1
        theta[i][(i+d-1)%d]=1
    M=np.dot(x,np.dot(theta,x))
    e=np.exp(J*M)
    M=e*M
    return [e,M]

def calcM(J):
    a=[0,0]
    for i in range(pow(2,d)):
        x=deciToBy(i,d+2)
        b=f(J,x)
        a[0]+=b[0]
        a[1]+=b[1]
        m=a[1]/a[0]
        m=m-10#surch for zero point ot 'm-a0 '
    return m 
##############mcmc sampling##############
def mcmc_sampling(mt,J,x=[]):
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
        print(np.log(f(J,x)[0]/J),"#=f")
    return x
########## mcmc newto method ##########
def mcmc_newton(J):
    mt,ns,sd=1000, 200,3 # mc-time(until equilibrium), #sample, sampling distance 
    def genMCMC(x=[]):
        x0=[]
        l=len(x)
        for t in range(mt+ns*sd):
            for i in range(l):
                valu=(( x[(l+i-1)%l]+x[(i+1)%l] ))
                r=np.exp(-J*valu)/(np.exp(J*valu)+np.exp(-J*valu))
                R=np.random.uniform(0,1)
                if(R<=r):
                    x[i]=1
                else:
                    x[i]=-1
            if(t>mt and t%3==0):
                if(len(x0)==0):
                    x0=x
                else:
                    x0=np.vstack((x0,x))
        return x0.astype(int)                
    def calcf(J,x=[[]]):
        a=[0,0]
        for i in range(100):
            y=x[i,:]
            b=f(J,y)
            a[0]+=b[0]
            a[1]+=b[1]
            m=a[1]/a[0]
            m=m-10#surch for zero point ot 'm-a0 '
        return m
    x0=np.ones(d)# Initialize
    x=genMCMC(x0)# Generate mcmc-samples
    J0=root(calcf,0.1,args=x)
    print("J0=",J0)
    return J0##################main##################
#temp=calcM(0.28697343)
x=np.ones(d)
#plt.subplot(121)
#plt.imshow(x)
#y=mcmc_sampling(3000,10/d,x)
mcmc_newton(0.1)
#plt.subplot(122)
#plt.imshow(y)
#plt.show()

#print("m=",temp)
#J0=root(calcM,0.1)
#print(J0)
#print('solution=',J0.x)


#########################################


