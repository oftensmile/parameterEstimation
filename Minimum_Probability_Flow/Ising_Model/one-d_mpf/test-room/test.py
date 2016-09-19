import numpy as np
import sys
vec=np.random.uniform(0.0,1.0,16)
"""
print(vec)
fname="test-output.txt"
f=open(fname,"w")
f.write(str(100))
f.close()
"""
d=5
class Mytest:
    def __init__(self,x,y):
        self.x=x
        self.y=y

for i in range(5):
    if(i==0):
        array=[Mytest(i,10+i)]
    else:
        array=np.append(array,Mytest(i,10+i))

def convert_binary_to_decimal(x=[]):
    #supportce that entry of the x is -1 or 1.
    size_x=len(x)
    decimal=0
    for i in range(size_x):
        decimal+=int(0.5*(x[i]+1) * 2**i )
    return decimal

def convert_decimal_to_binary(x):
    y=np.zeros(d)
    b=x
    y[0]=int(2*(x%2-0.5))
    a=x%2
    for l in range(1,d):
        b=b-a*2**(l-1)
        a=np.sign(b%(2**(l+1)))
        y[l]=int(2*(a-0.5))
    return y

def find_neighbor(x):
    array=convert_decimal_to_binary(x)
    ne
    for l in range(d):
        array[l]*=-1
        index=convert_binary_to_decimal(array)

