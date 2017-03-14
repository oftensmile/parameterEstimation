#! /usr/bin/env python
#-*-coding:utf-8-*-
import random, math
import numpy as np

def gEnergy(mu,sig,x):
    return 0.5*(x-mu)**2/sig**2

def T_HB(mu,sig,x):
    x_prop = x + 0.1*np.random.normal()
    # prob to accept a new state
    Energ_x = gEnergy(mu,sig,x)
    Energ_x_prop = gEnergy(mu,sig,x_prop)
    p_of_x = 1.0 / (1.0 + np.exp(-(Energ_x-Energ_x_prop)))
    r = np.random.uniform(0,1)
    if (r<p_of_x):
        x = x_prop
    return x

def T_MP(mu,sig,x):
    x_prop = x + 0.1*np.random.normal()
    # prob to accept a new state
    Energ_x = gEnergy(mu,sig,x)
    Energ_x_prop = gEnergy(mu,sig,x_prop)
    p_of_x = np.exp(-(Energ_x_prop-Energ_x)) 
    r = np.random.uniform(0,1)
    if (r<p_of_x):
        x = x_prop
    return x

def GD_ML(a,b,eps,lr,s_mean,s_var):
    t,grad=0,1.0
    while (t<epc_max and abs(grad)>eps):
        da, db = (a-s_mean), (b - s_var)
        a = a - (lr/np.log(2+t))*da #ML
        #b = b - lr*(b - np.sum((s-a)**2)/(n-1))#ML
        b = b - (lr/np.log(2+t))*db#M
        grad = abs(da)+ abs(db)
        t+=1
    return (a,b)

def GD_CD(a,b,eps,lr,s=[]):
    t,grad=0,1.0
    a_vec, b_vec= [], []
    a_mean,b_mean = 0.0,0.0
    while (t<epc_max and abs(grad)>eps):
        s1 = []
        for x in s:
            #y = T_MP(a,np.sqrt(b),x)
            y = T_HB(a,np.sqrt(b),x)
            s1.append(y)
        s1_mean,s1_var=np.mean(s1),np.var(s1)
        da, db = (a-s1_mean), (b - s1_var)
        a = a - (lr/np.log(2+t))*da #ML
        #b = b - lr*(b - np.sum((s-a)**2)/(n-1))#ML
        b = b - (lr/np.log(2+t))*db #M
        # Smoothing: Simple Moveing Average (size L=30)
        a_vec.append(a), b_vec.append(b)
        # WHY 100 ??????
        if(len(a_vec)>100):
            a_vec.remove(a_vec[0]),b_vec.remove(b_vec[0])
            a_mean, b_mean = np.mean(a_vec), np.mean(b_vec)
            #print a, b, a_mean,b_mean
        grad = abs(a_mean-s1_mean)+ abs(b_mean-s1_var)
        t+=1
    #return (a_mean,b_mean)
    return (a,b)

if __name__ == '__main__':
    mu, sig = 1.0, 1.0
    flag = 0 
    n_list = [40,80,160,320,640,1280]
    
    fname="stav-"+str(M)+"-mu-"+str(mu)+"-sig"+str(sig)+"-cd-ML-0309.dat"
    #fname="stav-"+str(M)+"-mu-"+str(mu)+"-sig"+str(sig)+"-cd-MP-0309.dat"
    #fname="stav-"+str(M)+"-mu-"+str(mu)+"-sig"+str(sig)+"-cd-HB-0309.dat"
    f=open(fname,"w") 
    #n_list = [80,120,160,240,320,480,640,960,1280,1920,2560,3840,5120,7680]
    #n_list = [1000,1000,1000,1000,1000]
    M =1000
    lr, eps, epc_max = 1,0.00001, 1000
    for n in n_list:
        mu_vec, sig_vec = [], []
        for m in range(M):
            s = np.random.normal(mu,sig,n)
            s_mean,s_var = np.mean(s),np.var(s) 
            s_myvar = np.sum((s-s_mean)**2)/(n-1)
            ml_sol=[s_mean,np.sqrt(s_myvar)]
            a, b = mu+0.1, sig**2+0.1 # Initial guessing of mu and sigma**2
            t, grad =0,1.0
            # Iteratively solver = GD+ML.
            if(flag==0):
                a,b=GD_ML(a,b,eps,lr,s_mean,s_var)
            elif(flag==1):
                a,b=GD_CD(a,b,eps,lr,s)
            mu_vec.append(a), sig_vec.append(b)
        mean_mu, mean_sigma = np.mean(mu_vec)/np.sqrt(M), np.mean(sig_vec)/np.sqrt(M)
        print  n, mean_mu-mu, mean_sigma-sig**2
        f.write(str(n)+"  " + str(mean_mu-mu) + "  " + str((mean_sigma-sig**2)/np.sqrt(M)) + "\n")
        f.close()
