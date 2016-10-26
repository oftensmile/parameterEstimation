import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
import statistics as st
np.random.seed(0)
t_interval = 100
#parameter ( System )
d, N_sample = 8,500 #124, 1000
N_remove=100
def gen_mcmc(J,x=[[]]):
    for i1 in range(d):
        for i2 in range(d):
            #Heat Bath
            diff_E=2.0*J*x[i1][i2]*(x[(i1+d-1)%d][i2]+x[(i1+1)%d][i2]+x[i1][(i2+d-1)%d]+x[i1][(i2+1)%d])#E_new-E_old
            r=1.0/(1+np.exp(diff_E)) 
            R=np.random.uniform(0,1)
            if(R<=r):
                x[i1][i2]=x[i1][i2]*(-1)
    return x

def calc_E(J,x=[[]]):
    erg=0
    for i1 in range(d):
        for i2 in range(d):
            erg+=x[i1][i2]*(x[(i1+d-1)%d][i2]+x[(i1+1)%d][i2]+x[i1][(i2+d-1)%d]+x[i1][(i2+1)%d])
    return 0.5*erg

def get_E_C(J):
    #x=np.ones((d,d))
    x=np.random.choice([-1,1],(d,d))
    E,E2=0.0,0.0
    for n in range(N_sample+N_remove):
        if(n<N_remove):
            x=np.copy(gen_mcmc(J,x))
        else:
            for t in range(t_interval):
                x=np.copy(gen_mcmc(J,x))
            E_temp=calc_E(J,x)
            E+=E_temp/N_sample
            E2+=E_temp**2 / N_sample
    return (E,-E2)

if __name__ == '__main__':
    fname="J-dependency-on-E-C.dat"
    f=open(fname,"w")
    J_slice=np.arange(2.5,3.0,0.05) 
    for J in J_slice:
        E,C=get_E_C(J)
        print(J,E,C)
        f.write(str(J)+"  "+str(E)+"  "+str(C)+"\n")
