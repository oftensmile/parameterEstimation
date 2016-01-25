import numpy as np
import matplotlib.pyplot as  plt
sqrt = np.emath.sqrt
from scipy.optimize import fsolve
from scipy.optimize import root
np.random.seed(0)
d=10

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
        #print(np.log(f(J,x)[0]/J),"#=f")
    return x
#####Find inverse map by using linear regression method####)
# At first, learning shape of function
def regres_from_mcmc(k,nJ,Jmax,ns):#nJ=sample num
    dJ=Jmax/nJ
    J,PhiofJ=np.zeros(nJ),np.zeros(nJ)
    for j in range(nJ):
        x=np.ones(d)
        x=mcmc_sampling(500,dJ*j,x)
        m=0
        for t in range(ns):
            x=mcmc_sampling(3,dJ*j,x)
            m+=f(dJ*j,x)[1]
        m/=ns
        J[j],PhiofJ[j]=dJ*j,m
    z=np.polyfit(J,PhiofJ,k)
    return z
def root_phi(J,m,z=[]):
    p=np.poly1d(z)  # this may cannot use for root() function
    return p(J)-m
########## mcmc newto method ##########
def mcmc_newton(J,z=[]):
    mt,ns,sd=1000, 200,3 # mc-time(until equilibrium), #sample, sampling distance 
    #
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
    def root_h(h,c,tau=[],x=[[]]):
        for i in range(ns):
            b=h*np.dot(tau,np.array(x[i]))
        b/=d
        h=b-c
        return h
    def calcPhi(J,x=[[]]):
        a=[0,0]
        #size=len(x[0])
        for i in range(ns):
            b=f(J,x[i,:])#=[e,M]
            a[0]+=b[0]
            a[1]+=b[1]
        m=a[1]/a[0]
        return m
   
    x0=np.ones(d)# Initialize
    x=genMCMC(x0)# Generate mcmc-samples
    h0=root(root_h,0.1,(10,np.ones(d),x))#[1,..,1] is damaged sample
    m=calcPhi(J,x)
    J0=root(root_phi,0.1,(m,z))
    return J0
##################main##################
J0=1.0#initial guess
ns=100
#Learnign phi function
k,nJ,Jmax=20,100,10
z=regres_from_mcmc(k,nJ,Jmax,ns)

for q in range(50):
    m=0
    x=np.ones(d)#initial state
    x=mcmc_sampling(500,J0,x)
    for t in range(ns):
        x=mcmc_sampling(3,J0,x)
        m+=f(J0,x)[1]
    m/=ns
    #print("m=",m)
    J0=mcmc_newton(m,z)
    J0=np.asscalar(J0.x)
    print(J0,"#=J0")
#print(J0)



#temp=calcM(0.28697343)
#x=np.ones(d)
#plt.subplot(121)
#plt.imshow(x)
#y=mcmc_sampling(3000,10/d,x)
#mcmc_newton(0.1)
#plt.subplot(122)
#plt.imshow(y)
#plt.show()

