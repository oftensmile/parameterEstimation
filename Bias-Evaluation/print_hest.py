#! /usr/bin/env python
#-*-coding:utf-8-*-
import random, math

if __name__ == '__main__':
    N = 640
    dm=1.8/(N)
    for i in range(N):
        plot_m = -0.95 + dm*i  
        plot_h = 0.5 * math.log((1.0+plot_m)/(1.0-plot_m))
        print plot_m, plot_h 
