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

def cd1_single_update(parameter=[]):
    J,h = parameter 
    cost = 0.0
    global sample
    M = len(sample)
    grad_h,grad_J = 0.0, 0.0
    for s in sample:
        s1s2,s1as2=s[0]*s[1], s[0]+s[1]
        grad_J += -4*s1s2/(1.0+np.exp(2.0*J*s1s2+h*s1as2))
        grad_h += -2 *(s1as2/(1.0+np.exp(2.0*h*(s1as2))) + s1as2/(1.0+np.exp(2.0*J*s1s2+h*s1as2)))
    
    return (grad_J,grad_h) 

def alpha_two_single(parameter=[],*args):
    J,h = parameter
    alpha,q1,q4 = args
    if(J>h):
        cost_cd_two = cd1_of_J_geq_h(parameter,q1,q4)
    else:
        cost_cd_two = cd1_of_h_geq_J(parameter,q1,q4)
    global sample 
    cost_cd_sing = cd1_single_update(parameter)
    G0 =  alpha*cost_cd_two[0]+(1-alpha)*cost_cd_sing[0]
    G1 =  alpha*cost_cd_two[1]+(1-alpha)*cost_cd_sing[1]
    return (G0,G1) 

def convert_xx_xax_to_q1_q4(xx,xax):
    q1 = (xx+xax+1)/4.0
    q4 = (xx-xax+1)/4.0
    return q1,q4


if __name__ == '__main__':
    eps,lr,max_epc = 0.0000001, 1.0, 300
    h0,J0 =0.1, 0.5#0.2, 0.1
    #N_list = [40,80,160,320,640,1280,1920,2560,3840,5120,7680]
    #M_list = [1000,10000,100000]
    N_list = [320,640]
    M_list = [500]
    for M in M_list:
        fname="stav-"+str(M)+"-J0-"+str(J0)+"-h0-"+str(h0)+"-cd-sing-two-update-alpha2.dat"
        f=open(fname,"w")
        f.write("#N, bias_h, b_std_h/sqrt(M), bias_J,b_std_J/sqrt(M),..(cd1-two),..(cd1-single).\n")
        sqrtM = np.sqrt(M)
        for N in N_list:
            bh_list_cd = np.zeros(M)
            bJ_list_cd = np.zeros(M)
            bh_list_cd_sing = np.zeros(M)
            bJ_list_cd_sing = np.zeros(M)
            bh_list_cd_alpha025 = np.zeros(M)
            bJ_list_cd_alpha025 = np.zeros(M)
            bh_list_cd_alpha05 = np.zeros(M)
            bJ_list_cd_alpha05 = np.zeros(M)
            bh_list_cd_alpha075 = np.zeros(M)
            bJ_list_cd_alpha075 = np.zeros(M)
            bh_list_cd_alpha090 = np.zeros(M)
            bJ_list_cd_alpha090 = np.zeros(M)
            m = 0
            while(m<M):
                mean=mean_stat(h0,J0,N)
                xx,xax = mean[0],mean[1]
                if((1+xx+xax)!=0 and (1+xx-xax)!=0):
                    count=0
                    h_init, J_init = h0+0.1, J0+0.1
                    q1,q4 = convert_xx_xax_to_q1_q4(xx,xax)
                    args_cd=(q1,q4)
                    if(J0>h0):
                        solv_cd=optimize.root(cd1_of_J_geq_h,[J_init,h_init],args=args_cd).x 
                    elif(h0>J0):
                        solv_cd=optimize.root(cd1_of_h_geq_J,[J_init,h_init],args=args_cd).x
                    J_cd,h_cd=solv_cd[0],solv_cd[1] 
                    global sample 
                    solv_cd_sing =optimize.root(cd1_single_update,[J_init,h_init]).x 
                    J_cd_sing,h_cd_sing=solv_cd_sing[0],solv_cd_sing[1] 
                    args_alpha025,args_alpha05,args_alpha075,args_alpha090= (0.25,q1,q4),(0.5,q1,q4),(0.75,q1,q4),(0.999,q1,q4)
                    
                    solv_cd_alpha025 =optimize.root(alpha_two_single,[J_init,h_init],args=args_alpha025).x
                    J_cd_alpha025,h_cd_alpha025 = solv_cd_alpha025[0],solv_cd_alpha025[1]
                    solv_cd_alpha05 =optimize.root(alpha_two_single,[J_init,h_init],args=args_alpha05).x   
                    J_cd_alpha05,h_cd_alpha05 = solv_cd_alpha05[0],solv_cd_alpha05[1]
                    solv_cd_alpha075 =optimize.root(alpha_two_single,[J_init,h_init],args=args_alpha075).x   
                    J_cd_alpha075,h_cd_alpha075 = solv_cd_alpha075[0],solv_cd_alpha075[1]
                    solv_cd_alpha090 =optimize.root(alpha_two_single,[J_init,h_init],args=args_alpha090).x   
                    J_cd_alpha090,h_cd_alpha090 = solv_cd_alpha090[0],solv_cd_alpha090[1]
                    
                    bh_list_cd[m], bJ_list_cd[m] = h_cd-h0, J_cd-J0 
                    bh_list_cd_sing[m], bJ_list_cd_sing[m] = h_cd_sing-h0, J_cd_sing-J0
                    
                    bh_list_cd_alpha025[m],bJ_list_cd_alpha025[m] = h_cd_alpha025-h0, J_cd_alpha025-J0
                    bh_list_cd_alpha05[m],bJ_list_cd_alpha05[m] = h_cd_alpha05-h0, J_cd_alpha05-J0
                    bh_list_cd_alpha075[m],bJ_list_cd_alpha075[m] = h_cd_alpha075-h0, J_cd_alpha075-J0
                    bh_list_cd_alpha090[m],bJ_list_cd_alpha090[m] = h_cd_alpha090-h0, J_cd_alpha090-J0
                    m+=1
                #end if 
            bias_h_cd, b_std_h_cd = np.mean(bh_list_cd), np.std(bh_list_cd)/sqrtM
            bias_J_cd, b_std_J_cd = np.mean(bJ_list_cd), np.std(bJ_list_cd)/sqrtM
            bias_h_cd_sing, b_std_h_cd_sing = np.mean(bh_list_cd_sing), np.std(bh_list_cd_sing)/sqrtM
            bias_J_cd_sing, b_std_J_cd_sing = np.mean(bJ_list_cd_sing), np.std(bJ_list_cd_sing)/sqrtM
            
            bias_h_cd_alpha025, b_std_h_cd_alpha025 = np.mean(bh_list_cd_alpha025), np.std(bh_list_cd_alpha025)/sqrtM
            bias_J_cd_alpha025, b_std_J_cd_alpha025 = np.mean(bJ_list_cd_alpha025), np.std(bJ_list_cd_alpha025)/sqrtM
            bias_h_cd_alpha05, b_std_h_cd_alpha05 = np.mean(bh_list_cd_alpha05), np.std(bh_list_cd_alpha05)/sqrtM
            bias_J_cd_alpha05, b_std_J_cd_alpha05 = np.mean(bJ_list_cd_alpha05), np.std(bJ_list_cd_alpha05)/sqrtM
            bias_h_cd_alpha075, b_std_h_cd_alpha075 = np.mean(bh_list_cd_alpha075), np.std(bh_list_cd_alpha075)/sqrtM
            bias_J_cd_alpha075, b_std_J_cd_alpha075 = np.mean(bJ_list_cd_alpha075), np.std(bJ_list_cd_alpha075)/sqrtM
            bias_h_cd_alpha090, b_std_h_cd_alpha090 = np.mean(bh_list_cd_alpha090), np.std(bh_list_cd_alpha090)/sqrtM
            bias_J_cd_alpha090, b_std_J_cd_alpha090 = np.mean(bJ_list_cd_alpha090), np.std(bJ_list_cd_alpha090)/sqrtM
            f.write(str(N) 
         #           + "  " + str(abs(bias_h_cd)) + "  "  + str(b_std_h_cd) 
                    + "  " + str(abs(bias_J_cd)) + "  "  + str(b_std_J_cd)
         #           + "  " + str(abs(bias_h_cd_sing)) + "  "  + str(b_std_h_cd_sing) 
                    + "  " + str(abs(bias_J_cd_sing)) + "  "  + str(b_std_J_cd_sing)
         #           + "  " + str(abs(bias_h_cd_alpha025)) + "  "  + str(b_std_h_cd_alpha025) 
                    + "  " + str(abs(bias_J_cd_alpha025)) + "  "  + str(b_std_J_cd_alpha025)
         #          + "  " + str(abs(bias_h_cd_alpha05)) + "  "  + str(b_std_h_cd_alpha05) 
                    + "  " + str(abs(bias_J_cd_alpha05)) + "  "  + str(b_std_J_cd_alpha05)
         #          + "  " + str(abs(bias_h_cd_alpha075)) + "  "  + str(b_std_h_cd_alpha075) 
                    + "  " + str(abs(bias_J_cd_alpha075)) + "  "  + str(b_std_J_cd_alpha075)
         #          + "  " + str(abs(bias_h_cd_alpha090)) + "  "  + str(b_std_h_cd_alpha090) 
                    + "  " + str(abs(bias_J_cd_alpha090)) + "  "  + str(b_std_J_cd_alpha090)
                    +" #time="+ str(time.time()) +"\n" )
        f.close()    
