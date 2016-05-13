import numpy as np
from scipy.optimize import fsolve
sqrt = np.emath.sqrt


def f(x):
    d=5
    return ((2*np.cosh(x))**(d-1)+(2*np.sinh(x))**(d-1))/ ((2*np.cosh(x))**d+(2*np.sinh(x))**d)
x = fsolve(f, 1.0)
print(x)
