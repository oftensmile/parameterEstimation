#! /usr/bin/env python
#-*-coding:utf-8-*-
import random, math
import numpy as np
import time 
from scipy.optimize import fsolve
from scipy import linalg
import itertools
import time
#import warnings

def gen_prob(*args):
    h0,J0 = args
    d = len(h0)
    global Prob0, state , p0 
    p0 = np.zeros(2**d)
    Prob0 = np.zeros(2**d)
    z0,n_s = 0.0, 0
    for s in state:
        n_s = list_to_binary(s)
        ene0=calc_energy(h0,J0,s)
        # hist[0],hist[1] = state, Prob(state)
        p0[n_s] = np.exp(ene0)
        z0+=np.exp(ene0)
    p0= p0/z0 
    sum_P = 0.0
    
    for i in range(2**d):
        sum_P += p0[i]
        Prob0[i] = sum_P
    return 0 

def list_to_binary(a):
    d = len(a)
    a = (a+np.ones(d))/2
    a = np.array(a)
    n_a=a.dot(2**np.arange(a.size)[::-1])
    return int(n_a) 

def calc_energy(h=[],J=[],s=[]):
    temps = np.copy(s).tolist()
    head = temps.pop(0)
    temps.append(head)
    temps = np.array(temps)
    ss = np.array(s)*temps 
    energy = np.dot(h,np.array(s))+np.dot(J,np.array(ss)) 
    return energy

def gen_sample(N,h=[],J=[]):
    sample = []
    global Prob0, state
    for n in range(N):
        r = np.random.uniform()
        m = binary_search(Prob0,r)
        s = np.copy(state[m])
        sample.append(np.copy(s))
    return sample

def binary_search(array, target):
    lower = 0
    upper = len(array)
    while lower < upper:   # use < instead of <=
        x = lower + (upper - lower) // 2
        val = array[x]
        if target == val:
            return x
        elif target > val:
            if lower == x:   # this two are the actual lines
                break        # you're looking for
            lower = x
        elif target < val:
            upper = x
    return x

def empirical_dist(d,sample):
    prob = np.zeros(2**d) # d = num of spins.
    for s in sample:
        n_s = int(list_to_binary(s)) 
        prob[n_s]+=1.0
    prob = prob/len(sample)
    return prob

def calc_data_hJ(sample=[[]]):
    N = len(sample)
    sdata = np.ones(d)
    ssdata = np.ones(d)
    for s in sample:
        temps = np.copy(s).tolist()
        head = temps.pop(0)
        temps.append(head)
        temps = np.array(temps)
        ss = np.array(s)*temps 
        sdata =sdata  + s
        ssdata  = ssdata  + ss 
    sdata  = sdata /N
    ssdata = ssdata /N
    return (sdata,ssdata) 

def eq_of_ml(parameter=[],*args):
    d = int(len(parameter)/2.0)
    h,J = parameter[:d],parameter[d:] 
    sdata,ssdata=args
    s_ml,ss_ml = calc_model_mle(h,J)
    eq_s= sdata - s_ml
    eq_ss= ssdata - ss_ml
    para_out=np.append(eq_s,eq_ss)
    return para_out 

def calc_model_mle(h=[],J=[]):
    d = len(h)
    z=0
    s_model=np.zeros(d)
    ss_model=np.zeros(d)
    global state, ene_max
    ene_max = 0.0
    z = 0.0
    for s in state:
        temps = np.copy(s).tolist()
        head = temps.pop(0)
        temps.append(head)
        temps = np.array(temps)
        ss = np.array(s)*temps 
        exp_of = np.dot(ss,J)+np.dot(s,h)
        z += np.exp(exp_of)
        if(ene_max<exp_of):
            ene_max = exp_of
    z = z * np.exp(-ene_max)
    for s in state:
        temps = np.copy(s).tolist()
        head = temps.pop(0)
        temps.append(head)
        temps = np.array(temps)
        ss = np.array(s)*temps 
        exp_of = np.dot(ss,J)+np.dot(s,h)
        exp_s=np.exp(exp_of-ene_max)
        ss_model = ss_model + ss * exp_s /z 
        s_model = s_model + ss * exp_s / z 
    return (s_model,ss_model) 

def calc_model_cd1_HB_LSE(parameter=[],*args):
    global P0, state, ene_max
    d = int(len(parameter)/2.0)
    n_state = 2**d
    h,J = parameter[:d],parameter[d:] 
    h0,J0=args
    eq_of_cd1_h=np.zeros(d)
    eq_of_cd1_J=np.zeros(d)
    z0 = 0
    #ene_max=0.0
    #for s in state:
    #    energy = calc_energy(h,J,s)
    #    if(ene_max < energy):
    #        ene_max=energy
    for s in state:
        #ene0 = calc_energy(h0,J0,s)
        ene = calc_energy(h,J,s)
        n_s = int(list_to_binary(s)) 
        for i in range(d):
            si = np.copy(s)
            si1 = np.copy(s)
            si[i]*=-1
            si1[(i+1)%d]*=-1
            ene_i = calc_energy(h,J,si)
            ene_i1 = calc_energy(h,J,si1)
            if (P0[n_s]<(0.1/n_state)):
                A_si_s = 0.0
                A_si_s1 = 0.0
            else:
                A_si_s = np.exp(ene_i - ene_max) / (np.exp(ene) + np.exp(ene_i - ene_max))  * P0[n_s]
                A_si_s1 = np.exp(ene_i1 - ene_max) / (np.exp(ene - ene_max)+ np.exp(ene_i1 - ene_max))  * P0[n_s]
            eq_of_cd1_h[i]+=-2*s[i]*A_si_s
            eq_of_cd1_J[i]+=-2*s[i]*s[(i+1)%d]*(A_si_s + A_si_s1)
    para_out=np.append(eq_of_cd1_h,eq_of_cd1_J)
    return para_out 



if __name__ == '__main__':
    M=1 
    h0,J0 =[0.1,0.2,0.3],[0.1,0.0,0.1] 
    d = len(h0) 
    h_init,J_init= [0.1,0.1,0.1],[0.01,0.1,0.01]
    initial = (h_init,J_init)
    ans =np.append(h0,J0)
    global Prob0, state
    state= list(itertools.product([1,-1],repeat=d))
    gen_prob(h0,J0)
    #fname = "test-Bias-N-dependence-M"+str(n_statis)+"-d"+str(d)+"-5.dat"
    #f = open(fname, "w") 
    N_list = [10,50,100,500,1000,5000,10000,50000,100000]
    s_time = time.time()
    for N in N_list:
        ml,cd = np.zeros(2*d), np.zeros(2*d)
        """
        ml_h,ml_J = np.zeros(d), np.zeros(d)
        ml_hh,ml_JJ = np.zeros(d), np.zeros(d)
        cd_h,cd_J = np.zeros(d), np.zeros(d)
        cd_hh,cd_JJ = np.zeros(d), np.zeros(d)
        """
        for m in range(M):
            sample = gen_sample(N,h0,J0)
            global P0, p0 
            P0 = empirical_dist(d,sample)
            
            #sdata,ssdata = calc_data_hJ(sample)
            sdata,ssdata = calc_model_mle(h0,J0)
            #MLE
            solv_ml=fsolve(eq_of_ml,initial,args=(sdata,ssdata)) - ans
            #CD1
            solv_cd=fsolve(calc_model_cd1_HB_LSE,initial,args=(h0,J0)) - ans
            #print "solv_ml = \n", solv_ml
            #print "solv_cd = \n", solv_cd
            print N, sum(solv_ml), sum(solv_cd)
            print solv_ml
            print solv_cd_
            ml = ml + solv_ml
            cd = cd + solv_cd
        ml = ml/M
        cd = cd/M 
        #print N,", ML=", min(ml), max(ml), ", CD=", min(cd), max(cd)
"""
            ml_h = ml_h + solv_ml[:d]
            ml_hh = ml_hh + solv_ml[:d]**2
            ml_J = ml_J + solv_ml[d:]
            ml_JJ = ml_JJ + solv_ml[d:]**2
            #np.sqrt(sum((solv_ml[:d]-ans[:d])**2))/d, np.sqrt(sum((solv_ml[d:]-ans[d:])**2))/d
            cd_h = cd_h + solv_cd[:d]
            cd_hh = cd_hh + solv_cd[:d]**2
            cd_J = cd_J + solv_cd[d:]
            cd_JJ = cd_JJ + solv_cd[d:]**2
            print N,", ML=", min(ml_h), max(ml_h), min(ml_J), max(ml_J),", CD=", min(cd_h), max(cd_h), min(cd_J), max(cd_J)
"""



