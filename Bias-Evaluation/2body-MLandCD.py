#! /usr/bin/env python
#-*-coding:utf-8-*-
import random, math
import numpy as np
import time 
from scipy import optimize

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

def solv_hJ(xx,xandx):
    h = 0.25*np.log( (1+xx+xandx) / (1+xx-xandx) )
    J = 0.25*np.log( (1+xx+xandx)*(1+xx-xandx) / (1-xx)**2 )
    return (h,J)

def mleq_of_hJ(h,J,xx,xandx):
    ch,sh,eJ = np.cosh(2*h), np.sinh(2*h), np.exp(-2*J)
    mleq1 = xx- (ch-eJ) / (ch+eJ)
    mleq2 = xandx - 2*sh / (ch+eJ) 
    return (mleq1,mleq2)

def loss_func_cd_master(h,J,s=[]):
    n=len(s)
    loss_func=0.0
    for x in s:
        xx=x[0]*x[1]
        xax=x[0]+x[1]
        loss_func += np.log(np.exp(-2.0*h*xax)+1.0)/float(n)
        +2.0*np.log(np.exp(-2.0*J*xx-h*xax)+1.0)/float(n)
    return loss_func

def CD1_objective(t=[]):
    h,J = t
    cost = 0.0
    global sample
    M = len(sample)
    for s in sample:
        s1s2,s1as2=s[0]*s[1], s[0]+s[1]
        cost += np.log(1.0 + np.exp(-2.0*h*s1as2))/M+2.0*np.log(1.0+np.exp(-2.0*J*s1s2-h*s1as2))/M
    return cost 

def log_likelihood(t=[],*stat):
    h,J = t
    xax,xx = stat 
    return -(J*xx+h*xax-np.log(np.exp(J)*np.cosh(2.0*h)+np.exp(-J))) 

if __name__ == '__main__':
    h0,J0 =0.4, 0.8
    N_list = [40,80,240,480,960,1280,2560,5120,7680]
    M_list = [10000]
    #N_list = [10,100,500,1000]
    #M_list = [10]
    for M in M_list:
        fname="stav-"+str(M)+"-J0-"+str(J0)+"-h0-"+str(h0)+"-master.dat"
        f=open(fname,"w")
        f.write("#N, bias_h, bias_J, b_std_h/sqrt(M), b_std_J/sqrt(M)+ (...CD1...) \n")
        for N in N_list:
            bh_list = np.zeros(M)
            bJ_list = np.zeros(M)
            bh_cd_list = np.zeros(M)
            bJ_cd_list = np.zeros(M)
            #for m in range(M): 
            m = 0
            while(m<M):
                mean=mean_stat(h0,J0,N)
                global sample 
                if((1+mean[0]+mean[1])!=0 and (1+mean[0]-mean[1])!=0):
                    #h,J = solv_hJ(mean[0],mean[1]) 
                    stat = (mean[1],mean[0])
                    h,J = optimize.fmin_cg(log_likelihood,[0.1,0.1],args=stat)
                    h_cd,J_cd = optimize.fmin_cg(CD1_objective,[0.1,0.1])
                    bh_list[m], bJ_list[m] = h-h0, J-J0 
                    bh_cd_list[m], bJ_cd_list[m] = h_cd-h0, J_cd-J0 
                else:
                    m -=1
                m+=1
            bias_h, b_std_h = np.mean(bh_list), np.std(bh_list)
            bias_J, b_std_J = np.mean(bJ_list), np.std(bJ_list)
            bias_h_cd, b_std_h_cd = np.mean(bh_cd_list), np.std(bh_cd_list)
            bias_J_cd, b_std_J_cd = np.mean(bJ_cd_list), np.std(bJ_cd_list)
            f.write(str(N) + "  " 
                    + str(abs(bias_h)) + "  "  + str(b_std_h/np.sqrt(M)) 
                    + "  " + str(abs(bias_J)) + "  "  + str(b_std_J/np.sqrt(M))
                    + str(abs(bias_h_cd)) + "  "  + str(b_std_h_cd/np.sqrt(M)) 
                    + "  " + str(abs(bias_J_cd)) + "  "  + str(b_std_J_cd/np.sqrt(M))
                    +"\n" )
        f.close()    
