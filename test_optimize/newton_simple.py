import numpy as np
import scipy as scipy
from numpy.linalg import inv
import sys

N=100
epoc=100
def newton():
    v=np.array([4.0,4.0])
    eps=0.01
    H=np.reshape(np.array([2.0,-1.0,-1.0,2.0]),(2,2))
    Hinv=inv(H)
    def f(x,y):
        return x**2 - y*x +y**2
    def grad_f(v=[]):
        u1=2*v[0]-v[1]
        u2=2*v[1]-v[0]
        return np.array([u1,u2])

    for t in range(epoc):
        #v=v-0.1*np.dot(H,grad_f(v))
        v=v-0.1*np.dot(Hinv,grad_f(v))
        print(v[0],v[1],f(v[0],v[1]))

def newton2():
    v=np.array([4.0,4.0])
    eps=0.01
    def f(x,y):
        return x**2 - y*x +y**2 + 0.5*x**4
    def grad_f(v=[]):
        u1=2*v[0]-v[1]+0.2*v[0]**3
        u2=2*v[1]-v[0]
        return np.array([u1,u2])

    for t in range(epoc):
        H=np.reshape(np.array([2.0+0.6*v[0]**2,-1.0,-1.0,2.0]),(2,2))
        Hinv=inv(H)
        v=v-0.1*np.dot(Hinv,grad_f(v))
        print(v[0],v[1],f(v[0],v[1]))

if __name__ == '__main__':
    newton2()
