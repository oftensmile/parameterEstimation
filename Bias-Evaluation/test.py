#! /usr/bin/env python
#-*-coding:utf-8-*-
import random, math
import numpy as np
n,i=0,0
while(i<10):
    n += 1
    i += 1
    if (n%3!=0):
        print i,n
    else:
        print i,n, "b"
        i -= 1
        print i,n, "a"


