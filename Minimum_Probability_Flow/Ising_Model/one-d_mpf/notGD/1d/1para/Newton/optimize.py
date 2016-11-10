import numpy as np
import math
from scipy.optimize import fsolve
from scipy.optimize import root
d=3
def func(x):
    return x*np.cos(x-4)
x0=fsolve(func,0.0)
x0=fsolve(func, -0.74)
    
# calc partition func
"""
def f(d,x=[]):
    theta = np.zeros(d,d)
    for i in range(d):
        for j in range(d):
            if(j=i+1||i=j+i):
                theta[i][j]=1
    theta[d-1][0],theta[0][d-1]=1,1
    M=np.dot(x,theta*x)
    #print("M=",M)
    return M
"""
"""
# make all variey of configurations
def vec_f(x=[]):
    t0=x[0]**2+x[1]**2+x[2]**2
    t1=x[0]+x[1]+x[2]
    t2=x[0]**0.5+x[1]**0.5+x[2]**0.5
    return [t0,t1,t2]
x=[0.1,0.2,0.3]
sol = root(vec_f,x,method="lm")
print("my_sol=",sol)
my_sol=[ 1.52506095, -0.22261215, -0.2226235 ]
print("\n\n","check=",vec_f(my_sol))
"""
def mat_f(x=[]):
    t=np.zeros(d)
    for i in range(d):
        for j in range(d):
            t[i]=t[i]+x[i+d*j]
    return t
x0=np.random.random((d,d))
x0_vec=np.reshape(x0,(1,d*d))
print("x0_vec=",len(x0_vec))
print("x0_vec=",x0_vec[0])
print("mat=",mat_f(x0_vec[0]),"\n")
sol_mat=root(mat_f,x0_vec[0],method="krylov")
print("sol_mat",sol_mat)

