import numpy as np
import math
from scipy.optimize import fsolve
from scipy.optimize import root
from scipy.optimize import fmin_cg
from scipy.optimize import brent 
d=3
def func(x):
    return x*np.cos(x-4)
#x0=fsolve(func,0.0)
#x0=fsolve(func, -0.74)
    
def fun(x):
    return [0.5*(x[0]-x[1])**2-1.0,0.5*(x[1]-x[0])**2+x[1]]

def jac_fun(x):
    return np.array([
        [1 + 1.5 * (x[0] - x[1])**2,-1.5 * (x[0] - x[1])**2],
        [-1.5 * (x[1] - x[0])**2,1 + 1.5 * (x[1] - x[0])**2]])

def g_3d(x):
    a=[1.0,1.0,2.0]
    #return [(x[0]-a[0])**2,(x[1]-a[1])**2,(x[2]-a[2])**2]
    return np.array([(x[0]-a[0])**2,(x[1]-a[1])**2,(x[2]-a[2])**2])

def jac(x):
    return np.array([[2,0,0],[0,2,0],[0,0,2]])
x0=[0.1,0.1,0.1]
#sol=fmin_cg(g_3d,x0)
#sol=root(g_3d,[0,0,0],jac=jac,method="hybr")
#sol_root=root(fun,[0,0],jac=jac_fun,method="hybr")
sol=root(g_3d,[0,0,0],method="hybr")
sol_root=root(fun,np.zeros(2),method="hybr")
print("sol=",sol.x,"\n")
print("sol_root=",sol_root.x)







"""
x0=np.random.random((d,d))
x0_vec=np.reshape(x0,(1,d*d))
print("x0_vec=",len(x0_vec))
print("x0_vec=",x0_vec[0])
print("mat=",mat_f(x0_vec[0]),"\n")
sol_mat=root(mat_f,x0_vec[0],method="krylov")
print("sol_mat",sol_mat)
"""
