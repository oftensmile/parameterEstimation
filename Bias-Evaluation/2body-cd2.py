#! /usr/bin/env python
#-*-coding:utf-8-*-
import random, math
import numpy as np
import time 
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

def mleq_of_hJ(h,J,xx,xandx):
    ch,sh,eJ = np.cosh(2*h), np.sinh(2*h), np.exp(-2*J)
    mleq1 = xx- (ch-eJ) / (ch+eJ)
    mleq2 = xandx - 2*sh / (ch+eJ) 
    return (mleq1,mleq2)

if __name__ == '__main__':
    eps,lr,max_epc = 0.0000001, 1.0, 300
    h0,J0 =0.2, 0.1
    N_list = [40,80,160,320,640,1280,1920,2560,3840,5120,7680]
    #N_list = [40,80,160,500]
    M_list = [10000,100000]
    for M in M_list:
        fname="stav-"+str(M)+"-J0-"+str(J0)+"-h0-"+str(h0)+"-cd-0313.dat"
        f=open(fname,"w")
        f.write("#N, bias_h, bias_J, b_std_h/sqrt(M), b_std_J/sqrt(M) \n")
        for N in N_list:
            bh_list = np.zeros(M)
            bJ_list = np.zeros(M)
            m = 0
            while(m<M):
                mean=mean_stat(h0,J0,N)
                dh,dJ = 1, 1
                if((1+mean[0]+mean[1])!=0 and (1+mean[0]-mean[1])!=0):
                    count=0
                    h, J = h0+0.1, J0+0.1
                    while((abs(dh)+abs(dJ))>eps and count<max_epc):
                        dh,dJ = mleq_of_hJ(h,J,mean[0],mean[1])
                        alr=lr/(1.0+np.log(1.0+count))
                        h,J = h+alr*dh, J+alr*dJ
                        #if(abs(h)+abs(J)>100):print count,h,J
                        count+=1
                    bh_list[m], bJ_list[m] = h-h0, J-J0 
                    m+=1
                #end if 
            bias_h, b_std_h = np.mean(bh_list), np.std(bh_list)
            bias_J, b_std_J = np.mean(bJ_list), np.std(bJ_list)
            #print M, N, bias_h, bias_J, b_std_h, b_std_J 
            f.write(str(N) + "  " + str(abs(bias_h)) + "  "  + str(b_std_h/np.sqrt(M)) 
                    + "  " + str(abs(bias_J)) + "  "  + str(b_std_J/np.sqrt(M))
                    +"\n#time="+ str(time.time())  )
        f.close()    
