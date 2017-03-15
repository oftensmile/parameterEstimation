#! /usr/bin/env python
#-*-coding:utf-8-*-
import random, math
import numpy as np

def mean_x(h,N): 
    mean=0
    p = 1.0/(1+math.exp(-2.0*h))
    flag=True
    while(flag):
        for i in range(N):
            r = random.uniform(0,1)
            if (r<p):x=1
            elif (r>p):x=-1
            else: x = random.choice([-1,1]) 
            mean+=x
        mean/=float(N)
        if(abs(mean)<1):    flag=False
        elif(abs(mean)==1): mean=0
    return mean 

def propsal(K,*args):
    print "args@propsal=", args
    s,s_1m,s_1p,s_2m,s_2p=args
    d_E=-2*(-s)*(s_1m+s_1p+s_2m+s_2p) 
    p = np.exp(-K*d_E)
    r = np.random.uniform(0,1)
    s_prop=-s
    if(r>p):
        s_prop=s
    return s_prop


def MP_update(L,K,spins=[[]]):
    #random sorting is better than just simple update
    for i in range(L):
        for j in range(L):
            s = spins[i][j]
            s_1m,s_1p = spins[(i+L-1)%L][j],spins[(i+1)%L][j]
            s_2m,s_2p = spins[i][(j+L-1)%L],spins[i][(j+1)%L]
            s_set = (s,s_1m,s_1p,s_2m,s_2p)
            print "s_set@MP_update=", s_set
            s=propsal(K,s_set)
    return np.copy(spins) 

def specificheat(spins=[[]]):
    c1,c2=np.zeros(L),np.zeros(L)
    m1,m2=np.zeros(L),np.zeros(L)
    for i in range(L):
        for j in range(L):
            c1[i]+=spins[i][j]*spins[i][(j+1)%L]
            c2[i]+=spins[j][i]*spins[(j+1)%L][i]
            m1[i]+=spins[i][j]
            m2[i]+=spins[j][i]
    c1,c2,m1,m2 =c1/float(L),c2/float(L),m1/float(L),m2/float(L) 
    specificheat =(sum(c1-m1**2)+sum(c2-m2**2))/2.0 
    return specificheat

if __name__ == '__main__':
    K0 =2.0
    L=5
    spins=np.random.choice([-1,1],(L,L))
    T=1000
    for t in range(T):
        spins=MP_update(L,K0,spins)
        spheat=specificheat(spins)
        print specificheat

    
    
    
    #N_list = [10,20,30,40,80,120,160,240,320,480,640,960,1280,1920,2560,3840,5120,7680]
    #M_list = [10000,10000]
    #count = 0
    """for M in M_list:
        if(count==0):
            h0=0.1
        elif(count==1):
            h0=0.5
        fname="stav-"+str(M)+"-h0-"+str(h0)+".dat"
        f=open(fname,"w")
        for N in N_list:
            b_list = np.zeros(M)
            for m in range(M):
                mean=mean_x(h0,N)
                h = 0.5*math.log((1.0+mean)/(1.0-mean))
                #b_list[m] = h - h0
                b_list[m] = h  
            bias = np.mean(b_list)
            f.write( str(N) + "  " + str(bias) + "  "  + str(np.std(b_list)/np.sqrt(M)) +"\n" )
        f.close()
        count+=1
    """
