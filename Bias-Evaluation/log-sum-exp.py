#! /usr/bin/env python
#-*-coding:utf-8-*-
import numpy as np
import random, math
import itertools

def logsumexp(a,b):
    output=0
    if(a==b):
        output = a+np.exp(2)
    else:
        x,y = max(a,b),min(a,b)
        output = x + np.log( 1+np.exp(y-x) ) # = log( exp(x) + exp(y) )
    return output 

def calc_energy(h=[],J=[],s=[]):
    temps = np.copy(s).tolist()
    head = temps.pop(0)
    temps.append(head)
    temps = np.array(temps)
    ss = np.array(s)*temps 
    energy = np.dot(h,np.array(s))+np.dot(J,np.array(ss)) 
    return energy

def partition_conventional(h=[],J=[]):
    d = len(h)
    z=0
    state= list(itertools.product([1,-1],repeat=d))
    for s in state:
        exp_of=0.0
        for i in range(d):
            exp_of+=J[i]*s[i]*s[(i+1)%d]+h[i]*s[i]
        z += np.exp(exp_of)
    return z 

# This function is using logsumexp.
def partition(h=[],J=[]):
    d = len(h)
    z=0
    state = list(itertools.product([1,-1],repeat=d))
    energies = np.zeros(2**d) 
    i = 0
    for s in state:
        energies[i]=calc_energy(h,J,s) # inverse sign
        i+=1
    energies = sorted(energies,reverse=True) 
    a = logsumexp(energies[0],energies[1]) 
    for i in range(2,2**d):
        a = logsumexp(a,energies[(i)])
    return a

    return 0
if __name__ == '__main__':
    d=4
    h,J = np.ones(d)*0.1, np.ones(d)*0.1
    F_conventional = np.log(partition_conventional(h,J)) 
    F = partition(h,J)
    print "without logsumexp(): log(z) = ", F_conventional
    print "  using logsumexp(): log(z) = ", F










