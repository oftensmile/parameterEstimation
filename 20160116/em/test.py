import numpy as np
#from scipy.optimize import fsolve
sqrt = np.emath.sqrt
from scipy.optimize import fsolve
from scipy.optimize import root
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

def f(d,J,x=[]):
    theta = np.zeros((d,d),dtype=np.float)
    for i in range(d):
        for j in range(d):
            if(j==i+1 or i==j+i):
                theta[i][j]=1
    theta[d-1][0],theta[0][d-1]=1,1
    M=np.dot(x,np.dot(theta,x))
    e=np.exp(J*M)
    M=e*M
    return [e,M]

def calcM(J):
    a=[0,0]
    for i in range(pow(2,d)):
        x=deciToBy(i,d+2)
        b=f(d,J,x)
        a[0]+=b[0]
        a[1]+=b[1]
        a=a[1]/a[0]
        m=a-10#surch for zero point ot 'm-a0 '
        print("m",m)
    return m 
####################main##################
a=calcM(0.1)
print(a[1]/a[0],"#=[e,M]=",a)
J0=root(calcM,0)
print(J0)
#print("#J0=",J0," calcM(J0)=",calcM(J0.x))

#m+=f(6,0.1,x)
    #print(a)

