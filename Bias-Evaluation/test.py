#! /usr/bin/env python
#-*-coding:utf-8-*-
import random, math
import numpy as np

def myfunc(x):
    return x*x

def myfunc2(x,y):
    return x*y

#def myfunc2(x,y,a):
#    return x*y+a

a0 = 1.0
a = np.arange(10)
ab = [np.arange(10),2*np.arange(10)]

A = np.linspace(0.05,1.55,100)
i=0
for a in A:
    print i,a
    i+=1
