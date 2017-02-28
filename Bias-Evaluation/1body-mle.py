#! /usr/bin/env python
#-*-coding:utf-8-*-
import random, math
import numpy as np

def mean_x(h,N): 
    mean=0
    p = 1.0/(1+math.exp(-2.0*h))
    for i in range(N):
        r = random.uniform(0,1)
        if (r<p):x=1
        elif (r>p):x=-1
        else: x = random.choice([-1,1]) 
        mean+=x
    mean/=float(N)
    p1=(mean+1.0)/2
    return (mean,p,p1) 

if __name__ == '__main__':
    h0 = 0.1 
    M = 200 # number of statistical average
    N_list = [20,30,40,80,120,160,240,320,480,640,960,1280,1920,2560,3840,5120,7680,10240,15360]

    for N in N_list:
        h_mean = 0
        p1_mean = 0
        for m in range(M):
            hotta=mean_x(h0,N) 
            h = 0.5*math.log((1.0+hotta[0])/(1.0-hotta[0]))
            h_mean += h/M
            p1_mean += hotta[2]/M
        print N, abs(h_mean-h0), p1_mean, hotta[1]



