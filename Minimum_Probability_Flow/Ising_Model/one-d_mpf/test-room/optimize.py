import numpy as np
from scipy import optimize

def f(x):   # The rosenbrock function
    return .5*(1 - x[0])**2 + (x[1] - x[0]**2)**2

def fprime(x):
    return np.array((-2*.5*(1 - x[0]) - 4*x[0]*(x[1] - x[0]**2), 2*(x[1] - x[0]**2)))

print("answer=",optimize.fmin_cg(f, [2, 2], fprime=fprime))
print("\nanswer=",optimize.fmin_cg(f, [2, 2]))  
