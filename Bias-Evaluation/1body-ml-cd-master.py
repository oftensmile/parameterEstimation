#! /usr/bin/env python
#-*-coding:utf-8-*-
import random, math
import numpy as np
from scipy import optimize

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

def eq_cd1_master(h,sample=[]):
    grad_cd1=0.0
    for s in sample:
        grad_cd1+=(-2.0*s)/(np.exp(-2.0*h*s)+1.0)
    return grad_cd1

def eq_cd1(h,m_data,sample=[]):
    my_set = []
    for x in sample:
        #y = TProb_HB(x,h)
        y = TProb_MP(x,h)
        my_set.append(y)
    m_model = np.mean(my_set)
    return m_data - m_model

def eq_ml(h,m_data):
    return m_data-np.tanh(h)

if __name__ == '__main__':
    eps,lr = 0.0000001, 1.0
    h0 =0.1
    h_init=h0+0.1
    N_list = [40,80,160,320,640,1280,1920,2560,3840,5120,7680]
    M_list = [10000,100000]
    #N_list = [40,50]
    #M_list = [10,20]
    for M in M_list:
        fname="stav-"+str(M)+"-h0-"+str(h0)+"-cd1mast-cd1-ml.dat"
        f=open(fname,"w")
        f.write("# N, mean_cd1_mast,mean_cd1,mean_ml,std_cd1mast/sqrtM,std_cd1/sqrtM,std_ml/sqrtM \n")
        for N in N_list:
            cd1_mast_list = np.zeros(M)
            cd1_list = np.zeros(M)
            ml_list = np.zeros(M)
            for m in range(M):
                set_data=mean_x(h0,N)
                m_model,m_data=100, np.mean(set_data[1])
                if(abs(m_data)!=N):
                    args_cd1_master=(set_data[1])
                    args_cd1=(m_data,set_data[1])
                    args_ml = (m_data)
                    sol_cd1_mast = optimize.root(eq_cd1_master,h_init,args=args_cd1_master).x[0]
                    sol_cd1 = optimize.root(eq_cd1,h_init,args=args_cd1).x[0]
                    sol_ml = optimize.root(eq_ml,h_init,args=args_ml).x[0]
                    cd1_mast_list[m],cd1_list[m],ml_list[m] = sol_cd1_mast,sol_cd1,sol_ml
            m_cd1_mas= abs(np.mean(cd1_mast_list))-h0
            m_cd1 = abs(np.mean(cd1_list))-h0
            m_ml = abs(np.mean(ml_list))-h0
            sqtM=np.sqrt(M)
            std_cd1_mas = np.std(cd1_mast_list)/sqtM
            std_cd1 = np.std(cd1_list)/sqtM
            std_ml = np.std(ml_list)/sqtM
            f.write( str(N) 
                    + "  " + str(m_cd1_mas) + "  "  + str(std_cd1_mas) 
                    + "  " + str(m_cd1) + "  "  + str(std_cd1)
                    + "  " + str(m_ml) + "  "  + str(std_ml)
                    +"\n" )
        f.close()

