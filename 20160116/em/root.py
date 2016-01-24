import numpy as np 
from scipy.optimize import root
def func(x,d=[]):
    return d[1]*x+d[0]*np.cos(x)
p=[2,1]
sol = root(func,0.3,p)
print( sol.x)
print (sol.fun)
