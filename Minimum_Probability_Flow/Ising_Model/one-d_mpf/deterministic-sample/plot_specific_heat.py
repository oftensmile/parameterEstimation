#! /usr/bin/env python
#-*-coding:utf-8-*-
import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
np.random.seed(1)
d=512   #d must to be even number.
J=1.0
n_sample=300

def tr_Tk(J,k):
    return (2*np.cosh(J))**k + (2*np.sinh(J))**k

#Tk[0]=Tk[1][1],Tk[2][2], Tk[1]=Tk[1][2],Tk[2][1]
def Tk(J,k):
    l1=(2*np.cosh(J))**k
    l2=(2*np.sinh(J))**k
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
    return p

def get_sample(j):
    X=np.zeros(d)
    #p(+)=p(-)=1/2
    X[0]=np.random.choice([-1,1])
    for k in range(1,d):
        p = pofx_given_xprev(j,k,X[0],X[k-1])
        X[k]=gen_x_pofx(p)
    return X

def calc_E(x_sample):
    E=0
    for k in range(d):
        E+=x_sample[k]*x_sample[(k+1)%d]
    return E

if __name__ == '__main__':
    n_J,J_min,J_max=50,0.0,1.0
    dJ=(J_max-J_min)/n_J
    vec_E=np.zeros(n_J)
    vec_C=np.zeros(n_J)
    for i in range(n_J):
        j=J_min+i*dJ
        E,C=0.0,0.0
        for n in range(n_sample):
            Xn=get_sample(j)
            En=calc_E(Xn)
            E+=En/n_sample
            C+=(En**2)/n_sample
        C-=E**2
        vec_E[i]=E
        vec_C[i]=C
        print(j,E,C)
    
    plt.plot(vec_E)
    plt.title("E")
    plt.plot(vec_C)
    plt.title("C")
    plt.show()
