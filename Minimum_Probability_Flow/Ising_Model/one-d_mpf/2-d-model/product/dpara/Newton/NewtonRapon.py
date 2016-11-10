import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import minimize 
sqrt = np.emath.sqrt

def f(x):
    d=5
    #return ((2*np.cosh(x))**(d-1)+(2*np.sinh(x))**(d-1))/ ((2*np.cosh(x))**d+(2*np.sinh(x))**d)
    return  (x-2.9)*(x-7.1)

def f2(x,a):
    d=5
    #return ((2*np.cosh(x))**(d-1)+(2*np.sinh(x))**(d-1))/ ((2*np.cosh(x))**d+(2*np.sinh(x))**d)
    return  (x-2.9)*(x-a)


x_newton = fsolve(f, 1.0)
x_nelder_mead = minimize(f2,1.0,method="Nelder-Mead",args=(1.3))
x_powell = minimize(f, 1.0,method="Powell")
x_cg = minimize(f, 1.0,method="CG")
x_bfgs = minimize(f, 1.0,method="BFGS")
#x_newtonCG = minimize(f, 1.0,method="Newton-CG")

print("newton=",x_newton)
print("nelder=",x_nelder_mead)
print("powell=",x_powell)
print("cgmeth=",x_cg)
print("bfgsme=",x_bfgs)
#print("newtCG=",x_newtonCG)
