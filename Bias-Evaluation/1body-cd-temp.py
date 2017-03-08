#! /usr/bin/env python
#-*-coding:utf-8-*-
import random, math
import numpy as np
import time
def mean_x(h,N): 
    mean=0
    p = 1.0/(1+math.exp(-2.0*h))
    flag=True
    set_x = []
    while(flag):
        for i in range(N):
            r = random.uniform(0,1)
            if (r<p):x=1
            elif (r>p):x=-1
            else: x = random.choice([-1,1]) 
            mean+=x
            set_x.append(x)
        mean/=float(N)
        if(abs(mean)<1):    flag=False
        elif(abs(mean)==1): mean=0
    return (mean, set_x)


def gen_tran_state(h,flag,set_receive=[]):
    set_return = []
    for x in set_receive:
        if flag%2==0:
            y = TProb_HB(x,h)
        else:
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


def TProb_HB(x,h):
    p = 1.0/(1.0 +np.exp(-2.0*h))
    r = random.uniform(0,1)
    if(r<p):
        y = 1 
    else:
        y = -1 
    return y


if __name__ == '__main__':
    eps,lr,max_epc = 0.0000001, 0.01, 2000
    h0 =0.1
    #N_list = [80,120,160,240,320,480,640,960,1280,1920,2560,3840,5120,7680]
    N_list = [200]
    #M_list = [100000,100000,1000000,1000000]
    M_list = [2]
    flag = 0
    for M in M_list:
        if (flag%2==0):
            fname="stav-"+str(M)+"-h0-"+str(h0)+"-cd-HB.dat"
        elif(flag%2==1):
            fname="stav-"+str(M)+"-h0-"+str(h0)+"-cd-MP.dat"
        flag += 1
        f=open(fname,"w")
        for N in N_list:
            b_list = np.zeros(M)
            for m in range(M):
                set_data=mean_x(h0,N)
                m_model,m_data=100, np.mean(set_data[1])
                h=0.1
                count=0
                while(abs(m_model-m_data)>eps and count<max_epc):
                    set_model=gen_tran_state(h,flag,set_data[1])
                    m_model=np.mean(set_model)
                    h += lr * (m_data - m_model)
                    print count, h
                    count+=1
                    b_list[m] = h - h0 
                    bias = np.mean(b_list)
                f.write( str(N) + "  " + str(bias) + "  "  + str(np.std(b_list)/np.sqrt(M)) +"\n"+str(time.time()) )
        f.close()

