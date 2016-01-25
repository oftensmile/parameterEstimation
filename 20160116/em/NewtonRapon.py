import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import newton
sqrt = np.emath.sqrt


def f(x):
    d=5
    return ((2*np.cosh(x))**(d-1)+(2*np.sinh(x))**(d-1))/ ((2*np.cosh(x))**d+(2*np.sinh(x))**d)
def g(x,a):
    return x**2 +a*x +1
    #return x**2 + -2*x + 1
#a=[-2.0,1.1]
x0=newton(g,0.1,args=2)
print("x0=",x0)



#x = fsolve(f, 1.0)
#print(x)
