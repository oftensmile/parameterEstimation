import numpy as np 
from scipy.optimize import root
def func(x,a,d=[]):
    return d[1]*x+d[0]*np.cos(x)+a
p=[2,1]
sol = root(func,0.3,(2,p))
print( sol.x)
print (sol.fun)
