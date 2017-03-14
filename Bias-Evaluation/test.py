#! /usr/bin/env python
#-*-coding:utf-8-*-
import random, math
import numpy as np
import time 
from scipy import optimize

def myfunc(x):
    return x*x

def myfunc2(x,y):
    return x*y

def f(x=[],*args):
    u,v = x
    h,J = args 
    #return 0.5*u**2 + 0.5*(v-1.0)**2
    return h*u**2 + J*(v-J)**2
def gradf(x=[],*args):
    u,v = x
    h,J = args
    gu=h*u
    gv=J*(v-J)
    return np.asarray((gu,gv))

def log_likelihood(t=[],*sample):
    h,J = t 
    xax,xx = sample 
    return -(J*xx+h*xax-np.log(np.exp(J)*np.cosh(2.0*h)+np.exp(-J))) 

def grad_log_likelihood(t=[],*sample):
    h,J = t 
    xax,xx = sample 
    gh = xax-2.0*np.sinh(2.0*h)/(np.cosh(2.0*h)+np.exp(-2.0*J))  
    gJ = xx-(np.cosh(2.0*h)-np.exp(-2.0*J))/(np.cosh(2.0*h)+np.exp(-2.0*J)) 
    return np.array((gh,gJ))

def solv_hJ(sample=[]):
    xax,xx = sample 
    h = 0.25*np.log( (1+xx+xax) / (1+xx-xax) )
    J = 0.25*np.log( (1+xx+xax)*(1+xx-xax) / (1-xx)**2 )
    return (h,J)

def CD1_objective(t=[]):
    h,J = t
    cost = 0.0
    global sample
    M = len(sample)
    for s in sample:
        s1s2,s1as2=s[0]*s[1], s[0]+s[1]
        cost += np.log(1.0 + np.exp(-2.0*h*s1as2))/M+2.0*np.log(1.0+np.exp(-2.0*J*s1s2-h*s1as2))/M
    return cost 

#   Generating Functions
def partition(h,J):
    return 2 * ( np.exp(J)*np.cosh(2*h)+np.exp(-J) ) 

def mean_stat(h,J,N):
    Z = partition(h,J)
    p1,p2 = np.exp(J+2*h)/Z, np.exp(J-2*h)/Z 
    mean_xx,mean_xandx=0.0,0.0
    global sample
    sample = []
    for i in range(N):
        r = random.uniform(0.0,1.0)
        if(r<p1):
            x=[1,1]
        elif(p1<=r and r<p1+p2):
            x=[-1,-1]
        elif(p1+p2<=r and r<1):
            x=[-1,1]
        mean_xx += x[0] * x[1]
        mean_xandx += x[0] + x[1]
        sample.append(x)
    mean_xx /= float(N)
    mean_xandx /= float(N)
    #p1_emp = 0.25*(1+mean_xx+mean_xandx)
    #p2_emp = 0.25*(1+mean_xx-mean_xandx)
    return ( mean_xx,mean_xandx )

if __name__ == '__main__':
    t0 = np.asarray((0.0,0.0))
    #sample=(1.0,0.6)
    h0,J0,N = 0.2,1.0,10
    xx,xax = mean_stat(h0,J0,N)
    stat = (xax,xx)
    t1 = time.time()
    solv1 = optimize.fmin_cg(log_likelihood,t0,args=stat)
    t2 = time.time()
    #solv2 = solv_hJ(sample)
    global sample 
    #solv3 = optimize.fmin_cg(CD1_objective,t0)
    t3 = time.time()
    print "h0,J0=", h0,J0
    print "\n\n solv1=\n", solv1
    #print "solv2=\n", solv3
    #print "time(ml),time(cd1)=", t2-t1,t3-t2
