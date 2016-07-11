import numpy as np
import scipy as scipy 
import sys

epc_mac=100
eps=0.00001
x=10
for t in range(epc_mac):
    temp_x=x
    x=np.tanh(1.2*x)
    print(x)
    if(np.abs(x-temp_x)<eps):
        break 
