#! /usr/bin/env python
#-*-coding:utf-8-*-
import random, math
import numpy as np
import time 
from scipy import optimize

def f(x,*args):
    a,b,s=args
    return (x-a-b)**2

if __name__ == '__main__':
    x0=0.0
    s=[1,1,1]
    args=(2,3,s)
    root=optimize.root(f,x0,args=args)
    print root.x[0]
