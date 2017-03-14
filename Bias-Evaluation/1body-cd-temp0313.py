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
    eps,lr,max_epc = 0.000001, 0.1, 2000
    h0 =0.1
    N_list = [80,120,160,240,320,480,640,960,1280,1920,2560,3840,5120,7680]
    #N_list = [50,100,300,500,800,1000]
    M_list = [10000,10000,100000,100000]
    M_list = [10,10]
    flag = 0
    for M in M_list:
        if (flag%2==0):
            fname="stav-"+str(M)+"-h0-"+str(h0)+"-cd-HB-0313.dat"
        elif(flag%2==1):
            fname="stav-"+str(M)+"-h0-"+str(h0)+"-cd-MP-0313.dat"
        flag += 1
        f=open(fname,"w")
        for N in N_list:
            b_list = np.zeros(M)
            b_mean_list = np.zeros(M)
            for m in range(M):
                set_data=mean_x(h0,N)
                dh,dh_mean,m_data=1.0,1.0, np.mean(set_data[1])
                dh_vec,h_vec=[],[]
                h=h0+0.2
                count,t=0,0
                #while(abs(dh)>eps and count<max_epc):
                while(abs(dh_mean)>eps and count<max_epc):
                    set_model=gen_tran_state(h,flag,set_data[1])
                    dh = m_data - np.mean(set_model)
                    h += (lr/np.log(2+t)) * dh 
                    #print count, h-h0,dh 
                    dh_vec.append(dh),h_vec.append(h)
                    if(len(dh_vec)>100):
                        dh_vec.pop(0),h_vec.pop(0)
                        dh_mean=np.mean(dh_vec)
                        h_mean=np.mean(h_vec)
                    count+=1
                b_list[m] = h - h0 
                b_mean_list[m] = h_mean - h0 
                bias = np.mean(b_list)
                bias_mean = np.mean(b_mean_list)
                #print h-h0,h_mean -h0 ,dh_mean, bias, bias_mean
            f.write( str(N) + "  " + str(bias) + "  "  + str(np.std(b_list)/np.sqrt(M)) + "  " + str(bias_mean) + "  "  + str(np.std(b_mean_list)/np.sqrt(M))+"\n")
        f.write("#"+str(time.time()) )
        f.close()

