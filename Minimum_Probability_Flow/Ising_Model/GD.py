#2016/05/21
import sys
import numpy as np
np.random.seed(0)
#parameter
eps=0.000001 #criterion of convergence
lr=10000.0    #learning rate
T=100000
def df(x):
    return 4.0*x**3-3.0*2.0*x
#f(x)=x*x*(x-1)*(x+1)
def f(x):
    return x**4-3.0*x**2
#input
print("#Put an initial value.")
input_line=sys.stdin.readline()
xt=1.0*float(input_line)
print("#t,xt,delta")
for t in range(T):
    f_old=f(xt)
    xt=xt-eps*df(xt)
    f_new=f(xt)
    delta=np.abs(f_old-f_new)
    diff=np.abs(-2.0-f_new)
    #print(t,"",xt,"",f_old,"",f_new)
    print(t,"",xt,"",delta,"",f_new)
    if(delta<eps):
        break


