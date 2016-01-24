######################################################
#   This programming intend to find hyper parameters
#   one dimension 
#   p({tau},{xi}|J,h)=exp(h*Sum_{tau_i*xi_i}+J*Sum_{xi_i*xi_j})/{Z(h)Z(J)}
#   find phi(^-1)(<xi_i*xi_j>), arctanh
####################################################
import numpy as np 
from scipy.optimize import fsolve
from scipy.optimize import root
from scipy import linalg
import time
np.random.seed(1)
d=8#number of spin variables
h,J=2,1
original,dameged=np.ones(d),2*np.array(np.random.random_integers(0,1,d)-0.5)
#######################SETUP############################
def xixj(original=[]):
    a,L=0,len(original)
    for i in range(L):
        a+=(original[i]*original[(i+1)%L])
    return a

#calc cost func(n.n interaction)
def cost_func(h,J,original=[], dameged=[]):
    L=len(original)
    cost_func=J*xixj(original)
    cost_func+=h*np.dot(original, dameged)
    return cost_func/L

#acquire spin config from equil pdf
def GibbsSampling(mcLength,h,J,original=[],dameged=[]):
    for t in range(mcLength):
        for i in range(len(original)):
            a=h*dameged[i]+J*(original[(i-1+d)%d]+original[(i+1)%d])
            r=np.exp(-a)/(np.exp(-a)+np.exp(a))#prob to accept original[i]=1
            R=np.random.uniform(0,1)
            if(R<=r):
                original[i]=1
            else:
                original[i]=-1

def f(x,p=[]):#p=[d,b]
    c=((2*np.cosh(x))**(p[0]-1)+(2*np.sinh(x))**(p[0]-1))/ ((2*np.cosh(x))**p[0]+(2*np.sinh(x))**p[0])-p[1]
    c= float(c)
    print("#b=",b,", c=",c)
    return c

def g(x):
    d,b=32,52
    c=((2*np.cosh(x))**(d-1)+(2*np.sinh(x))**(d-1))/ ((2*np.cosh(x))**d+(2*np.sinh(x))**d)-b
    c= float(c)
    return c
#####################INFERENCE##########################    
h_est,J_est=3,3
EMrepeat,thermAv=100,100
inference=np.ones(d)
p=[0.0,0.0]
for t in range(EMrepeat):
    #E-step 
    GibbsSampling(100, h_est, J_est,inference,dameged)#SSE
    a,b=0,0
    for i in range(thermAv):
        GibbsSampling(3,h_est,J_est,inference,dameged)
        a+=np.dot(inference,dameged)/d
        b+=xixj(inference) 
    a/=thermAv
    b/=thermAv
    #M-step 
    h_est=np.arctanh(a)
    p[0],p[1]=d,b
    #J_est=fsolve(f,1.0,args=p) 
    p = np.array(p, dtype=np.float)
    J_est=root(g,0.1)
    print("#h_est=",h_est)
GibbsSampling(100,h,J,original,dameged)
