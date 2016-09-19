import numpy as np
import time 
from scipy import linalg
import matplotlib.pyplot as plt
import csv 
np.random.seed(10)
t_interval = 50
d, N_sample =8,30 #124, 1000
N_remove = 100
lr,eps =0.1, 1.0e-100
t_gd_max=200 
#   E:= - \sum_{l}x_l x_{l+1} \theta_l
class State:
    def __init__(self,index, degree,state=[]):
        self.index=index    #This is not sorted order, but the decimal value of it.
        self.degree=degree
        self.state=np.copy(state)

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

def build_state_dist(X_sample=[[]]):
    global state_dist
    global address
    #state_dist = tuple(State(i,0,convert_decimal_to_binary(i)) for i in range(2**(d)))
    i=0
    address=[]
    for xn in X_sample:
        index=convert_binary_to_decimal(xn)
        if(len(address)==0):
            state_dist=[State(index,1,np.copy(xn))]
            address=np.append(address,index)
        elif(len(address)>0 and len(np.where(address==index)[0])==0):
            state_dist=np.append(state_dist, State(index,1,np.copy(xn)) )
            address=np.append(address,index)
        elif(len(address)>0 and len(np.where(address!=index)[0])>0):
            index2=np.where(address==index)[0][0]
            state_dist[index2].degree+=1
        i+=1
def gen_mcmc(J,x=[]):
    for i in range(d):
        diff_E=2.0*x[i]*J*(x[(i+1)%d]+x[(i+d-1)%d])
        r=1.0/(1+np.exp(diff_E)) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x

def prob_x_goout(th,j,x=[]):
    pout=1.0/(1.0 + np.exp( 2.0*th*x[j]*(x[(j+1)%d]+x[(j+d-1)%d]))) / d
    #   This must shape must be propotional to r in gen_mcmc
    return  pout

def prob_x_into(th,j,x=[]):
    pin=1.0/(1.0 + np.exp( -2.0*th*x[j]*(x[(j+1)%d]+x[(j+d-1)%d]))) / d
    return pin 
 
J_true=10
x = np.random.choice([-1,1],d)
for n in range(N_sample+N_remove):
    for t in range(t_interval):
        x = np.copy(gen_mcmc(J_true,x))
    if(n==N_remove):
        x_new=np.copy(x)
        X_sample = x_new
    elif(n>N_remove):
        x_new=np.copy(x)
        X_sample=np.vstack((X_sample,x_new))
build_state_dist(X_sample)
print(X_sample)
size_of_sample=len(state_dist)
bins=0.025
theta_slice=np.arange(.0,2.0,bins)
MPF_of_th=0

#theta_slice=[0]
#sum_degree=0
#for l in range(size_of_sample):
    #print(l,"",state_dist[l].degree,"",sum_degree)
    #sum_degree+=state_dist[l].degree

for th in theta_slice:
    MPF_of_th_old=MPF_of_th
    MPF_of_th=0.0
    prob1=0
    #Ihave to use only 
    for l in range(size_of_sample):
        xl=np.copy(state_dist[l].state)
        p_comin,p_goout=0.0,0.0
        index_xl=convert_binary_to_decimal(xl)
        for j in range(d):
            xl[j]*=-1
            index_neighbor=convert_binary_to_decimal(xl)
            size_neighbor=len(np.where(address[0]==index_neighbor)[0]) 
            if(size_neighbor>0):
                id2=np.where(address[0]==index_neighbor)[0][0] 
                p_comin+=state_dist[id2].degree*prob_x_into(th,j,xl)/d
            xl[j]*=-1
            p_goout+=prob_x_goout(th,j,xl)/d
        prob1_l=(p_comin+state_dist[l].degree*(1-p_goout))/N_sample
        prob1+=prob1_l
        if(prob1>0):#state_dist[l].degree must be positive value.
            MPF_of_th+=np.log(prob1*state_dist[l].degree/N_sample)/N_sample
    del_MPF = (MPF_of_th-MPF_of_th_old)/bins
    print(th,MPF_of_th,del_MPF,prob1)
