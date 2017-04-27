#! /usr/bin/env python
#-*-coding:utf-8-*-
import random, math
import numpy as np
import time 
from scipy.optimize import fsolve
from scipy import linalg
import itertools
#import warnings
#warnings.filterwarnings('ignore', 'The iteration is not making good progress')
PI = math.pi

def list_to_binary(a):
    d = len(a)
    a = (a+np.ones(d))/2
    a = np.array(a)
    n_a=a.dot(2**np.arange(a.size)[::-1])
    return n_a

def logsumexp(a,b):
    output=0
    if(a==b):
        output = a+np.exp(2)
    else:
        x,y = max(a,b),min(a,b)
        output = x + np.log( 1+np.exp(y-x) ) # = log( exp(x) + exp(y) )
    return output 

def calc_model_mle(h=[],J=[]):
    d = len(h)
    z=0
    s_model=np.zeros(d)
    ss_model=np.zeros(d)
    state= list(itertools.product([1,-1],repeat=d))
    for s in state:
        exp_of=0.0
        for i in range(d):
            exp_of+=J[i]*s[i]*s[(i+1)%d]+h[i]*s[i]
        exp_s=np.exp(exp_of)
        for i in range(d):
            ss_model[i]+=s[i]*s[(i+1)%d]*exp_s
            s_model[i]+=s[i]*exp_s
        z += exp_s
    s_model = s_model/ z
    ss_model = ss_model/ z
    return (s_model,ss_model) 

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

def gen_sample(N,h=[],J=[]):
    sample = []
    list_index=np.arange(d)
    t_min_max=300 #  maximum of minimum relaxation time
    s = np.random.choice([-1,1],d)
    t_wait= 50
    for t in range(t_min_max+t_wait*N+1):# or changing of the energy is smaller than epsilon. 
    # sequential update
        for i in list_index:
            r = random.uniform(0.0,1.0)
            p = np.exp(-2*(J[(i+d-1)%d]*s[i]*s[(i+d-1)%d]+J[(i+1)%d]*s[i]*s[(i+1)%d]+h[i]*s[i]))
            if(r<p):
                s[i]*=-1
        np.random.shuffle(list_index)
        if(t>t_min_max and (t-t_min_max)%t_wait==0):
            sample.append(np.copy(s))
    return sample

def empirical_dist(d,sample):
    prob = np.zeros(2**d)
    for s in sample:
        n_s = int(list_to_binary(s)) 
        prob[n_s]+=1
    prob = prob/len(sample)
    return prob

def calc_model_cd1_HB_LSE(parameter=[],*args):
    global P0 
    d = int(len(parameter)/2.0)
    h,J = parameter[:d],parameter[d:] 
    alpha, h0,J0=args
    state = list(itertools.product([1,-1],repeat=d))
    eq_of_cd1_h=np.zeros(d)
    eq_of_cd1_J=np.zeros(d)
    z0 = 0
    normalize_noise = 0.0
    ene_max=0.0
    for s in state:
        energy = calc_energy(h,J,s)
        ene0 = calc_energy(h0,J0,s)
        z0 += np.exp(ene0)
        if(ene_max < energy):
            ene_max=energy
    for s in state:
        ene0 = calc_energy(h0,J0,s)
        ene = calc_energy(h,J,s)
        n_s = int(list_to_binary(s)) 
        for i in range(d):
            si = np.copy(s)
            si1 = np.copy(s)
            si[i]*=-1
            si1[(i+1)%d]*=-1
            ene_i = calc_energy(h,J,si)
            ene_i1 = calc_energy(h,J,si1)
            A_si_s = np.exp(ene_i-ene_max) / (np.exp(ene-ene_max)+ np.exp(ene_i - ene_max))  * (alpha*np.exp(ene0)/z0 + (1.0-alpha)*P0[n_s])
            A_si_s1 = np.exp(ene_i1-ene_max) / (np.exp(ene-ene_max)+ np.exp(ene_i1 - ene_max)) * (alpha*np.exp(ene0)/z0 + (1.0-alpha)*P0[n_s])
            #A_si_s = np.exp(ene_i+ene0) / (np.exp(ene)+ np.exp(ene_i))
            eq_of_cd1_h[i]+=-2*s[i]*A_si_s
            eq_of_cd1_J[i]+=-2*s[i]*s[(i+1)%d]*(A_si_s + A_si_s1)
            eq_of_cd1_J[(i+d-1)%d]+=-2*s[i]*s[(i+1)%d]*A_si_s
    para_out=np.append(eq_of_cd1_h,eq_of_cd1_J)
    return para_out 

def calc_energy(h=[],J=[],s=[]):
    temps = np.copy(s).tolist()
    head = temps.pop(0)
    temps.append(head)
    temps = np.array(temps)
    ss = np.array(s)*temps 
    energy = np.dot(h,np.array(s))+np.dot(J,np.array(ss)) 
    return energy

def eq_of_ml(parameter=[],*args):
    d = int(len(parameter)/2.0)
    h,J = parameter[:d],parameter[d:] 
    alpha,sdata0,ssdata0,sdata,ssdata=args
    s_ml,ss_ml = calc_model_mle(h,J)
    eq_s= alpha*sdata0 + (1-alpha)*sdata - s_ml
    eq_ss= alpha*ssdata0 + (1-alpha)*ssdata - ss_ml
    para_out=np.append(eq_s,eq_ss)
    return para_out 

def eq_of_cd(parameter=[],*args):
    d = int(len(parameter)/2.0)
    h,J = parameter[:d],parameter[d:] 
    sdata,ssdata=args
    s_cd,ss_cd = calc_model_cd1_HB_LSE(h,J)
    para_out=np.append(sdata-s_cd, ssdata-ss_cd)
    return para_out 

if __name__ == '__main__':
    N,M=100, 2 
    #h,J = np.ones(d) * 0.2,np.ones(d) * 0.2 
    #h0,J0 =[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],[0.1,0.0,0.1,0.0,0.1,0.0,0.1,0.0,0.1] 
    h0,J0 =[0.1,0.2,0.3],[0.1,0.0,0.1] 
    d = len(h0)
    h_init,J_init= [0.1,0.1,0.1],[0.01,0.1,0.01] 
    #h_init,J_init= [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1] 
    d = len(h0) 
    initial = (h_init,J_init)
    ans =np.append(h0,J0)
    alpha_list = [1.0,0.8,0.6,0.4,0.25]
    #alpha_list = [1.0,0.999,0.995,0.9925,0.99,0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.5,0.01,0.0]
    f = open("average-alpha-dependence.dat", "w") 
        
    sdata0,ssdata0= calc_model_mle(h0,J0)
    for alpha in alpha_list:
        solve_ml_mean=np.zeros(len(h0)*2)
        solve_cd_mean=np.zeros(len(h0)*2)
        solve_ml_var=np.zeros(len(h0)*2)
        solve_cd_var=np.zeros(len(h0)*2)
        for m in range(M):
            sample = gen_sample(N,h0,J0)
            global P0 
            P0 = empirical_dist(d,sample)
            # this is sample from true parameter, and used for only cd method.
            sdata,ssdata = calc_data_hJ(sample)
            #MLE
            solve_ml=fsolve(eq_of_ml,initial,args=(alpha,sdata0,ssdata0,sdata,ssdata))
            solve_ml_mean=solve_ml_mean + (solve_ml-ans)/(M) 
            solve_ml_var=solve_ml_var + ((solve_ml-ans)/(M))**2
            #CD1
            solve_cd=fsolve(calc_model_cd1_HB_LSE,initial,args=(alpha,h0,J0))
            solve_cd_mean=solve_cd_mean + (solve_cd-ans)/(M)
            solve_cd_var=solve_cd_var + ((solve_cd-ans)/(M))**2

            diff_ml=solve_ml-ans
            diff_cd=solve_cd-ans
        solve_ml_var = solve_ml_var - solve_ml_mean**2
        solve_cd_var = solve_cd_var - solve_cd_mean**2
        f.write(str(alpha)+"  "+str(sum(np.abs(solve_ml_mean)/d))+"  "+str(sum(np.abs(solve_ml_var)/d))+"  "+str(sum(np.abs(solve_cd_mean)/d))+"  "+str(sum(np.abs(solve_cd_var)/d))+"\n")
    f.close()
