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

if __name__ == '__main__':
    h0, M =0.1, 30 # number of statistical average
    N_list = [10,20,30,40,80,120,160,240,320,480,640,960,1280,1920,2560,3840,5120,7680]
    #N_list = [20,30,40,80,120,160,240,320]
    for N in N_list:
        b_list = np.zeros(M)
        for m in range(M):
            mean=mean_x(h0,N)
            h = 0.5*math.log((1.0+hotta[0])/(1.0-hotta[0]))
            b_list[m] = h - h0
        bias = np.mean(b_list)
        print N, abs(bias), np.std(b_list)/np.sqrt(M)


