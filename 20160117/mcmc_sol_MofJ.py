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
def mcmc_sampling(J,x=[]):
    mt0,mt,l=3000,3,len(x)
    test=0
    for t in range(mt0):
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
##################main##################
temp=calcM(0.28697343)
x=np.ones(d)
#plt.subplot(121)
#plt.imshow(x)
y=mcmc_sampling(10/d,x)
#plt.subplot(122)
#plt.imshow(y)
#plt.show()

#print("m=",temp)
#J0=root(calcM,0.1)
#print(J0)
#print('solution=',J0.x)





#########################################
