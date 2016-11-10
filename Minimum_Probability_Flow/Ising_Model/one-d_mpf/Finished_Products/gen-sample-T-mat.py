#! /usr/bin/env python
#-*-coding:utf-8-*-
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
np.random.seed(0)
d=526    #d must to be even number.
J=1.0
#n_sample=32

def tr_Tk(J,k):
    return (2*np.cosh(J))**k + (2*np.sinh(J))**k

#Tk[0]=Tk[1][1],Tk[2][2], Tk[1]=Tk[1][2],Tk[2][1]
def Tk(J,k):
    l1=(2*np.cosh(J))**k
    l2=(2*np.sinh(J))**k
    print("l1,l2 = ", l1, l2)
    return ( 0.5*(l1+l2) , 0.5*(l1-l2) )

#p(x_i=+1|x_1-i)
def gen_x_pofx(p_value):
    r=np.random.uniform()
    if(p_value>r):x_prop=1
    else:x_prop=-1
    return x_prop

def pofx_given_xprev(J,k,x_1,x_prev):
    ind_plus_prev=int(0.5*(1-x_prev)) #if same sign=>0
    ind_first_prev=int(0.5*(1-x_1*x_prev)) #if same sign=>0
    p=Tk(J,1)[ind_plus_prev] * Tk(J,d-k)[0] / Tk(J,d-k+1)[ind_first_prev]
    print("(plus*prev,firs*prev,p,Tk)=",ind_plus_prev,ind_first_prev,p,Tk(J,d-k+1)[ind_first_prev])
    return p

if __name__ == '__main__':
    X=np.zeros(d)
    test_P=np.zeros(d)
    #p(+)=p(-)=1/2
    X[0]=np.random.choice([-1,1])
    for k in range(1,d):
        p = pofx_given_xprev(J,k,X[0],X[k-1])
        test_P[k]=p
        X[k]=gen_x_pofx(p)

    plt.plot(X)
    plt.title("config")
    plt.plot(test_P)
    plt.title("Prob")
    plt.show()
 
