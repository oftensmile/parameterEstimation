#! /usr/bin/env python
#-*-coding:utf-8-*-
import random, math
import numpy as np

def gEnergy(mu,sig,x):
    return 0.5*(x-mu)**2/sig**2

def T_HB(mu,sig,x):
    x_prop = x + rnadom.normal()
    # prob to accept a new state
    Energ_x = gEnergy(mu,sig,x)
    Energ_x_prop = gEnergy(mu,sig,x_prop)
    p_of_x = 1.0 / (1.0 + np.exp(-(Energ_x-Energ_x_prop)))
    r = np.random.uniform(0,1)
    if (r<p_of_x):
        x = x_prop
    return x

def T_MP(mu,sig,x):
    x_prop = x + rnadom.normal()
    # prob to accept a new state
    Energ_x = gEnergy(mu,sig,x)
    Energ_x_prop = gEnergy(mu,sig,x_prop)
    p_of_x = np.exp(-(Energ_x_prop-Energ_x)) 
    r = np.random.uniform(0,1)
    if (r<p_of_x):
        x = x_prop
    return x



if __name__ == '__main__':
    mu, sig = 1.0, 1.0
    n_list = [1000, 10000]
    lr, eps, epc_max = 1.0, 0.00001, 100000
    for n in n_list:
        s = np.random.normal(mu,sig,n)
        s_mean = np.mean(s)
        s_var = np.var(s) 
        s_myvar = np.sum((s-s_mean)**2)/(n-1)
        ml_sol=[s_mean,np.sqrt(s_myvar)]
        a, b = 2.0, 2.0 # Initial guessing of mu and sigma**2
        t = 0
        grad = 1.0
        while (t<epc_max and abs(a)>eps):
            da, db = (a-s_mean), (b - s_var)
            a = a - lr*da #ML
            #b = b - lr*(b - np.sum((s-a)**2)/(n-1))#ML
            b = b - lr*db#M
            grad = abs(da)+ abs(db)
            t+=1
        print n,t, a-mu, b-sig**2


