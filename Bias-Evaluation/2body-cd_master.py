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
    mean_xx /= float(N)
    mean_xandx /= float(N)
    return ( mean_xx,mean_xandx )

def gen_tran_state(h,set_receive=[]):
    set_return = []
    for x in set_receive:
        #y = TProb_HB(x,h)
        y = TProb_MP(x,h)
        set_return.append(y)
    return set_return 

def TProb_MP(x,h):
    y = - x
    p = np.exp(-(-h)*(y-x))
    r = random.uniform(0,1)
    if(r<p):
        y = -x
    else:
        y = x
    return y

def TProb_HB(h):
    p = 1.0/(1.0 +np.exp(-2.0*h))
    r = random.uniform(0,1)
    if(r<p):
        y = 1 
    else:
        y = -1 
    return y

def solv_hJ(xx,xandx):
    h = 0.25*np.log( (1+xx+xandx) / (1+xx-xandx) )
    J = 0.25*np.log( (1+xx+xandx)*(1+xx-xandx) / (1-xx)**2 )
    return (h,J)

def mleq_of_hJ(parameter=[],*args):
    J,h = parameter
    xx,xandx=args
    ch,sh,eJ = np.cosh(2*h), np.sinh(2*h), np.exp(-2*J)
    mleq1 = xx- (ch-eJ) / (ch+eJ)
    mleq2 = xandx - 2*sh / (ch+eJ) 
    return (mleq1,mleq2)

def cd1_of_J_geq_h(parameter=[],*args):
    J,h = parameter
    q1,q4=args
    exp_J_plus_h=np.exp(-2*(J+h))
    exp_J_minu_h=np.exp(-2*(J-h))
    exp_4h=np.exp(-4*h)
    cd1_xx=-4.0+4*(1+exp_J_plus_h)*q1+4*(1+exp_J_minu_h)*q4
    cd1_xax=4*(exp_J_plus_h+exp_4h)*q1-4*(exp_J_minu_h+1)*q4
    return (cd1_xx,cd1_xax)

def cd1_of_h_geq_J(parameter=[],*args):
    J,h = parameter
    q1,q4=args
    exp_J_plus_h=np.exp(-2*(J+h))
    exp_J_minu_h=np.exp(-2*(J-h))
    exp_4h=np.exp(-4*h)
    cd1_xx=-2*(1+1.0/exp_J_minu_h)+2*(1+1.0/exp_J_minu_h+2*exp_J_plus_h)*q1+2*(3+1.0/exp_J_minu_h)*q4
    cd1_xax=-2*(1-1.0/exp_J_minu_h)+2*(1+2*exp_4h-1.0/exp_J_minu_h+2*exp_J_plus_h)*q1-2*(3+1.0/exp_J_minu_h)*q4
    return (cd1_xx,cd1_xax)

def convert_xx_xax_to_q1_q4(xx,xax):
    q1 = (xx+xax+1)/4.0
    q4 = (xx-xax+1)/4.0
    return q1,q4

if __name__ == '__main__':
    eps,lr,max_epc = 0.0000001, 1.0, 300
    h0,J0 =0.2, 0.1
    N_list = [40,80,160,320,640,1280,1920,2560,3840,5120,7680]
    M_list = [10000,100000]
    #N_list = [40,80,160]
    #M_list = [10,100]
    for M in M_list:
        fname="stav-"+str(M)+"-J0-"+str(J0)+"-h0-"+str(h0)+"-cd-MP-master-0314.dat"
        f=open(fname,"w")
        f.write("#N, bias_h, b_std_h/sqrt(M), bias_J,b_std_J/sqrt(M),..(cd1).. \n")
        sqrtM = np.sqrt(M)
        for N in N_list:
            bh_list_ml = np.zeros(M)
            bJ_list_ml = np.zeros(M)
            bh_list_cd = np.zeros(M)
            bJ_list_cd = np.zeros(M)
            m = 0
            while(m<M):
                mean=mean_stat(h0,J0,N)
                xx,xax = mean[0],mean[1]
                if((1+xx+xax)!=0 and (1+xx-xax)!=0):
                    count=0
                    h_init, J_init = h0+0.1, J0+0.1
                    q1,q4 = convert_xx_xax_to_q1_q4(xx,xax)
                    args_ml,args_cd=(xx,xax),(q1,q4)
                    solv_ml=optimize.root(mleq_of_hJ,[J_init,h_init],args=args_ml).x 
                    J_ml,h_ml=solv_ml[0],solv_ml[1] 
                    if(J0>h0):
                        solv_cd=optimize.root(cd1_of_J_geq_h,[J_init,h_init],args=args_cd).x 
                    elif(h0>J0):
                        solv_cd=optimize.root(cd1_of_h_geq_J,[J_init,h_init],args=args_cd).x
                    J_cd,h_cd=solv_cd[0],solv_cd[1] 
                    bh_list_ml[m], bJ_list_ml[m] = h_ml-h0, J_ml-J0 
                    bh_list_cd[m], bJ_list_cd[m] = h_cd-h0, J_cd-J0 
                    m+=1
                #end if 
            bias_h_ml, b_std_h_ml = np.mean(bh_list_ml), np.std(bh_list_ml)/sqrtM
            bias_J_ml, b_std_J_ml = np.mean(bJ_list_ml), np.std(bJ_list_ml)/sqrtM
            bias_h_cd, b_std_h_cd = np.mean(bh_list_cd), np.std(bh_list_cd)/sqrtM
            bias_J_cd, b_std_J_cd = np.mean(bJ_list_cd), np.std(bJ_list_cd)/sqrtM
            f.write(str(N) 
                    + "  " + str(abs(bias_h_ml)) + "  "  + str(b_std_h_ml) 
                    + "  " + str(abs(bias_J_ml)) + "  "  + str(b_std_J_ml)
                    + "  " + str(abs(bias_h_cd)) + "  "  + str(b_std_h_cd) 
                    + "  " + str(abs(bias_J_cd)) + "  "  + str(b_std_J_cd)
                    +" #time="+ str(time.time()) +"\n" )
        f.close()    
