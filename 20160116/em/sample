import numpy as np
import math
from scipy.optimize import fsolve
d=8
def func(x):
    return x*np.cos(x-4)
x0=fsolve(func,0.0)
x0=fsolve(func, -0.74)
# calc partition func

def f(d,x=[]):
    theta = np.zeros(d,d)
    for i in range(d):
        for j in range(d):
            if(j=i+1||i=j+i):
                theta[i][j]=1
    theta[d-1][0],theta[0][d-1]=1,1
    M=np.dot(x,theta*x)







print(x0)

