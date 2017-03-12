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

def log_likelihood(t=[]):
    global stat_ml
    h,J = t
    xax,xx = stat_ml
    return -(J*xx+h*xax-np.log(np.exp(J)*np.cosh(2.0*h)+np.exp(-J))) 

def merge_like_cd(t=[]):
    global stat_merge
    alpha, xax,xx = stat_merge 
    cost_cd = CD1_objective(t)
    cost_ml = log_likelihood(t)
    G =  alpha*cost_cd+(1-alpha)*cost_ml
    return G 

if __name__ == '__main__':
    h0,J0 =0.1, 0.5
    #N_list = [40,80,120,160,240,320,480,640,960,1280,1920,2560,3840,5120,7680]
    #M_list = [10000]
    #N_list = [1000]
    N = 1000
    M_list = [1000]
    alpha_list = np.linspace(0.0,1.0,11)
    for M in M_list:
        fname="stav-"+str(M)+"-J0-"+str(J0)+"-h0-"+str(h0)+"-alpha.dat"
        f=open(fname,"w")
        f.write("#alpha, bias_h, bias_J, b_std_h/sqrt(M), b_std_J/sqrt(M), (...Mix_using_alpha...) total_col=13\n")
        for alpha in alpha_list:
            bh_list = np.zeros(M)
            bJ_list = np.zeros(M)
            bh_al_list = np.zeros(M)
            bJ_al_list = np.zeros(M)
            bh_diff_ml = np.zeros(M)
            bJ_diff_ml = np.zeros(M)
            m = 0
            while(m<M):
                mean=mean_stat(h0,J0,N)
                global sample 
                if((1+mean[0]+mean[1])!=0 and (1+mean[0]-mean[1])!=0):
                    global stat_ml, stat_merge
                    stat_ml = (mean[1],mean[0])
                    stat_merge = (alpha,mean[1],mean[0])
                    h,J = optimize.fmin_cg(log_likelihood,[0.0,0.0])
                    h_al,J_al = optimize.fmin_cg(merge_like_cd,[0.0,0.0])
                    bh_list[m], bJ_list[m] = h-h0, J-J0 
                    bh_al_list[m], bJ_al_list[m] = h_al-h0, J_al-J0 
                    bh_diff_ml[m], bJ_diff_ml[m] = h_al-h, J_al-J
                else:
                    m -=1
                m+=1
            bias_h, b_std_h = np.mean(bh_list), np.std(bh_list)
            bias_J, b_std_J = np.mean(bJ_list), np.std(bJ_list)
            bias_h_al, b_std_h_al = np.mean(bh_al_list), np.std(bh_al_list)
            bias_J_al, b_std_J_al = np.mean(bJ_al_list), np.std(bJ_al_list)
            bias_h_diff, b_std_h_diff = np.mean(bh_diff_ml), np.std(bh_diff_ml)
            bias_J_diff, b_std_J_diff = np.mean(bJ_diff_ml), np.std(bJ_diff_ml)
            f.write(str(alpha) + "  " 
                    + str(abs(bias_h)) + "  "  + str(b_std_h/np.sqrt(M)) 
                    + "  " + str(abs(bias_J)) + "  "  + str(b_std_J/np.sqrt(M))
                    + "  "+ str(abs(bias_h_al)) + "  "  + str(b_std_h_al/np.sqrt(M)) 
                    + "  " + str(abs(bias_J_al)) + "  "  + str(b_std_J_al/np.sqrt(M))
                    + "  "+ str(abs(bias_h_diff)) + "  "  + str(b_std_h_diff/np.sqrt(M)) 
                    + "  " + str(abs(bias_J_diff)) + "  "  + str(b_std_J_diff/np.sqrt(M))
                    +"\n" )
        f.close()   

