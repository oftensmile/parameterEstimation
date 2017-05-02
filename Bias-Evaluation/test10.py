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
#warnings.filterwarnings('ignore', 'The iteration is not making good progress')
PI = math.pi

def list_to_binary(a):
    d = len(a)
    a = (a+np.ones(d))/2
    a = np.array(a)
    n_a=a.dot(2**np.arange(a.size)[::-1])
    return int(n_a) 

def calc_model_mle(h=[],J=[]):
    d = len(h)
    z=0
    s_model=np.zeros(d)
    ss_model=np.zeros(d)
    state= list(itertools.product([1,-1],repeat=d))
    exp_of_max = 0.0
    z = 0.0
    for s in state:
        temps = np.copy(s).tolist()
        head = temps.pop(0)
        temps.append(head)
        temps = np.array(temps)
        ss = np.array(s)*temps 
        exp_of = np.dot(ss,J)+np.dot(s,h)
        z += np.exp(exp_of)
        if(exp_of_max<exp_of):
            exp_of_max = exp_of
    z = z * np.exp(-exp_of_max)
    for s in state:
        temps = np.copy(s).tolist()
        head = temps.pop(0)
        temps.append(head)
        temps = np.array(temps)
        ss = np.array(s)*temps 
        exp_of = np.dot(ss,J)+np.dot(s,h)
        #exp_of=0.0
        #for i in range(d):
        #    exp_of+=J[i]*s[i]*s[(i+1)%d]+h[i]*s[i]
        exp_s=np.exp(exp_of-exp_of_max)
        ss_model = ss_model + ss * exp_s / z
        s_model = s_model + ss * exp_s / z
        #for i in range(d):
        #    ss_model[i]+=s[i]*s[(i+1)%d]*exp_s
        #    s_model[i]+=s[i]*exp_s
        #z += exp_s
    #s_model = s_model/ z
    #ss_model = ss_model/ z
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
    global prob0, state
    for n in range(N):
        r = np.random.uniform()
        m = binary_search(prob0,r)
        s = np.copy(state[m])
        sample.append(np.copy(s))
    return sample

def empirical_dist(d,sample):
    prob = np.zeros(2**d) # d = num of spins.
    for s in sample:
        n_s = int(list_to_binary(s)) 
        prob[n_s]+=1.0
    prob = prob/len(sample)
    return prob

# Using modified log-sum-exp method. 
# The empirical is arleady an equilibrium.. 
def calc_model_cd1_HB_LSE(parameter=[],*args):
    global alpha,P0
    d = int(len(parameter)/2.0)
    n_state = 2**d
    h,J = parameter[:d],parameter[d:] 
    h0,J0=args
    state = list(itertools.product([1,-1],repeat=d))
    eq_of_cd1_h=np.zeros(d)
    eq_of_cd1_J=np.zeros(d)
    z0 = 0
    normalize_noise = 0.0
    ene_max=0.0
    g_expect = np.zeros(2*d)
    for s in state:
        energy = calc_energy(h,J,s)
        if(ene_max < energy):
            ene_max=energy
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

def gen_prob(*args):
    h0,J0 = args
    d = len(h0)
    state= list(itertools.product([1,-1],repeat=d))
    global prob0 
    hist0 = np.zeros(2**d)
    prob0 = np.zeros(2**d)
    z0,n_s = 0.0, 0
    for s in state:
        n_s = list_to_binary(s)
        ene0=calc_energy(h0,J0,s)
        # hist[0],hist[1] = state, Prob(state)
        hist0[n_s] = np.exp(ene0)
        z0+=np.exp(ene0)
    hist0= hist0/z0 
    sum_P = 0.0
    
    for i in range(2**d):
        sum_P += hist0[i]
        prob0[i] = sum_P
    return 0 

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

if __name__ == '__main__':
    N,n_statis=100 ,3 
    #h,J = np.ones(d) * 0.2,np.ones(d) * 0.2 
    h0,J0 =[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],[0.1,0.0,0.1,0.0,0.1,0.0,0.1,0.0,0.1] 
    #h0,J0 =[0.1,0.2,0.3,0.4],[0.1,0.0,0.1,0.0] 
    # This function does not need to convert
    d = len(h0) 
    global prob0, state
    gen_prob(h0,J0)
    state= list(itertools.product([1,-1],repeat=d))
    h_init,J_init= [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1] 
    #h_init,J_init= [0.1,0.1,0.1,0.1],[0.01,0.1,0.01,0.1] 
    initial = (h_init,J_init)
    ans =np.append(h0,J0)
    # this is sample from true parameter, and used for only cd method.
    fname = "test-Bias-N-dependence-M"+str(n_statis)+"-d"+str(d)+"-5.dat"
    f = open(fname, "w") 
    print "# N, ml_mean_h,ml_std_h, ml_mean_J,ml_std_J, cd_mean_h,cd_std_h,cd_mean_J,cd_std_J"
    #N_list = [20,30,40,60,80,160,320, 640, 960, 1280]
    N_list = [20,30,40,60]
    s_time = time.time()
    
    for N in N_list:
        ml_list_h, ml_list_J = np.zeros(n_statis), np.zeros(n_statis)
        cd_list_h, cd_list_J= np.zeros(n_statis), np.zeros(n_statis)
        ml_h,ml_J = np.zeros(d), np.zeros(d)
        ml_hh,ml_JJ = np.zeros(d), np.zeros(d)
        cd_h,cd_J = np.zeros(d), np.zeros(d)
        cd_hh,cd_JJ = np.zeros(d), np.zeros(d)
        for m in range(n_statis):
            sample = gen_sample(N,h0,J0)
            global alpha, P0 
            alpha = 0.0
            P0 = empirical_dist(d,sample)
            sdata,ssdata = calc_data_hJ(sample)
            #NOTE sdata0,ssdata0 are not used.
            sdata0,ssdata0= sdata,ssdata# calc_model_mle(h0,J0)
            #MLE
            solv_ml=fsolve(eq_of_ml,initial,args=(alpha,sdata0,ssdata0,sdata,ssdata))
            #CD1
            solv_cd=fsolve(calc_model_cd1_HB_LSE,initial,args=(h0,J0))

            ml_h = ml_h + solv_ml[:d]
            ml_hh = ml_hh + solv_ml[:d]**2
            ml_J = ml_J + solv_ml[d:]
            ml_JJ = ml_JJ + solv_ml[d:]**2
            #np.sqrt(sum((solv_ml[:d]-ans[:d])**2))/d, np.sqrt(sum((solv_ml[d:]-ans[d:])**2))/d
            cd_h = cd_h + solv_cd[:d]
            cd_hh = cd_hh + solv_cd[:d]**2
            cd_J = cd_J + solv_cd[d:]
            cd_JJ = cd_JJ + solv_cd[d:]**2
            
            #diff_cd_J = np.sqrt(sum((solv_cd[:d]-ans[:d])**2))/d, np.sqrt(sum((solv_cd[d:]-ans[d:])**2))/d
            #ml_list_h[m],ml_list_J[m] = diff_ml_h,diff_ml_J
            #cd_list_h[m],cd_list_J[m] = diff_cd_h,diff_cd_J

        ml_hh, ml_JJ = ((ml_hh/n_statis)+np.array(h0)**2-2*np.array(h0)*(ml_h/n_statis)) ,((ml_JJ/n_statis)+np.array(J0)**2-2*np.array(J0)*(ml_J/n_statis)) 
        ml_h, ml_J = ml_h/n_statis-h0, ml_J/n_statis-J0
        cd_h, cd_J = ( (cd_hh/n_statis)+np.array(h0)**2-2*np.array(h0)*(cd_h/n_statis)), ((cd_JJ/n_statis)+np.array(J0)**2-2*np.array(J0)*(cd_J/n_statis)) 
        cd_h, cd_J = cd_h/n_statis-h0, cd_J/n_statis-J0
        
        ml_mean_h,ml_std_h = np.mean(ml_h), np.sqrt(np.mean(ml_hh))
        ml_mean_J,ml_std_J = np.mean(ml_J),np.sqrt(np.mean(ml_JJ)) 
        cd_mean_h,cd_std_h = np.mean(cd_h),np.sqrt(np.mean(cd_hh))
        cd_mean_J,cd_std_J = np.mean(cd_J),np.sqrt(np.mean(cd_JJ))
        #ml_vari[m],cd_vari = diff_ml,diff_cd
        print  N, ml_mean_h,ml_std_h, ml_mean_J,ml_std_J, cd_mean_h,cd_std_h,cd_mean_J,cd_std_J
        f.write( str(N)+"  "+str(ml_mean_h)+"  " +str(ml_std_h) + "  " + str(ml_mean_J)+ "  " + str(ml_std_J) +"  "+str(cd_mean_h)+"  " +str(cd_std_h) + "  " + str(cd_mean_J)+ "  " + str(cd_std_J) +"\n")

    f_time = time.time()
    f.write("#computational time="+str(f_time-s_time))
    f.close()


