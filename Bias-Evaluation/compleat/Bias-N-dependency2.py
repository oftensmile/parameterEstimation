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

# This function is used as the expectations of data, which close to the equilibrium.
def calc_model_mle_with_noise(h0=[],J0=[]):
    global alpha, mu
    d = len(h0)
    z0=0
    s_model=np.zeros(d)
    ss_model=np.zeros(d)
    state= list(itertools.product([1,-1],repeat=d))
    normalize_noise = 0.0
    for s in state:
        noise=np.exp(-0.5*np.dot((mu-s),(mu-s)))
        normalize_noise+=noise
        ene0=calc_energy(h0,J0,s)
        z0+=np.exp(ene0)
    for s in state:
        noise=np.exp(-0.5*np.dot((mu-s),(mu-s)))
        ene0=calc_energy(h0,J0,s)
        exp_s=np.exp(ene0-np.log(z0))
        for i in range(d):
            ss_model[i] += s[i]*s[(i+1)%d] * ( alpha*exp_s + (1.0-alpha)*noise/normalize_noise )
            s_model[i] += s[i]* ( alpha*exp_s + (1.0-alpha)*noise/normalize_noise )
    s_model = s_model
    ss_model = ss_model
    return (s_model,ss_model) 

def calc_mean_covariance(h=[],J=[]):
    d = len(h)
    z=0
    mu=np.zeros(d)
    cov=np.zeros((d,d))
    state= list(itertools.product([1,-1],repeat=d))
    for s in state:
        s = np.array(s) 
        exp_of=0.0
        for i in range(d):
            exp_of+=J[i]*s[i]*s[(i+1)%d]+h[i]*s[i]
        exp_s=np.exp(exp_of)
        z += exp_s
        mu = mu + s * exp_s
        smat= np.matrix(s)
        cov = cov + np.dot(smat.T,smat) * exp_s
    mu = mu/ z
    cov = cov/ z - np.dot(np.matrix(mu).T,np.matrix(mu))  
    return (mu,cov) 

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

# Using modified log-sum-exp method. 
# The empirical is arleady an equilibrium.. 
def calc_model_cd1_HB_LSE(parameter=[],*args):
    global alpha,P0 
    d = int(len(parameter)/2.0)
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
        ene0 = calc_energy(h0,J0,s)
        z0 += np.exp(ene0)
        if(ene_max < energy):
            ene_max=energy
        ss_temp = np.copy(s).tolist()
        s_0 = ss_temp.pop(0)
        ss_temp.append(s_0)
        ss = np.array(ss_temp) * s 
        g = np.append(s,ss)
        g_expect =g_expect + g * np.exp(ene0)
    g_expect = g_expect / z0 
    #expect_p_r_of_g = np.zeros(2*d) 
    #expect_prim_s = np.zeros((2*d,2*d)) 
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

def calc_model_cd1_MP_LSE(parameter=[],*args):
    global P0, ene_max
    d = int(len(parameter)/2.0)
    state= list(itertools.product([1,-1],repeat=d))
    n_state = 2**d
    h,J = parameter[:d],parameter[d:] 
    h0,J0=args
    eq_of_cd1_h=np.zeros(d)
    eq_of_cd1_J=np.zeros(d)
    z0 = 0
    for s in state:
        ene = calc_energy(h,J,s)
        n_s = int(list_to_binary(s)) 
        for i in range(d):
            si = np.copy(s)
            si1 = np.copy(s)
            si[i]*=-1
            si1[(i+1)%d]*=-1
            ene_i = calc_energy(h,J,si)
            ene_i1 = calc_energy(h,J,si1)
            # sign(ene_i-ene) > 0 => 1
            # sign(ene_i-ene) < 0 => exp[(ene_i-ene)] 
            del_E = ene_i-ene
            del_E1 = ene_i-ene
            if (P0[n_s]<(0.1/n_state)):
                A_si_s = 0.0
                A_si_s1 = 0.0
            else:
                if(del_E<0):
                    A_si_s = np.exp(del_E)
                else:
                    A_si_s = 1 
                
                if(del_E1<0):
                    A_si_s1 = np.exp(del_E1)
                else:
                    A_si_s1 = 1 
                #A_si_s = np.exp(ene_i - ene_max) / (np.exp(ene) + np.exp(ene_i - ene_max))  * P0[n_s]
                #A_si_s = np.exp(ene_i - ene_max) / (np.exp(ene) + np.exp(ene_i - ene_max))  * P0[n_s]
                #A_si_s1 = np.exp(ene_i1 - ene_max) / (np.exp(ene - ene_max)+ np.exp(ene_i1 - ene_max))  * P0[n_s]
            eq_of_cd1_h[i]+=-2*s[i]*A_si_s * P0[n_s] 
            eq_of_cd1_J[i]+=-2*s[i]*s[(i+1)%d]*(A_si_s + A_si_s1) * P0[n_s] 
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
    N,n_statis=100,1 
    #h,J = np.ones(d) * 0.2,np.ones(d) * 0.2 
    #h0,J0 =[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],[0.1,0.0,0.1,0.0,0.1,0.0,0.1,0.0,0.1] 
    h0,J0 =[0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1] 
    # This function does not need to convert
    #mu,Sigma = calc_mean_covariance(h0,J0)  # Sima is not used.
    d = len(h0) 
    #h_init,J_init= [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1] 
    h_init,J_init= [0.01,0.1,0.01,0.1,0.01],[0.01,0.1,0.01,0.1,0.01] 
    initial = (h_init,J_init)
    ans =np.append(h0,J0)
    # this is sample from true parameter, and used for only cd method.
    fname = "Bias-N-dependence-M"+str(n_statis)+"-d"+str(d)+"-0511-2.dat"
    f = open(fname, "w") 
    print("# N, ml_mean_h,ml_std_h, ml_mean_J,ml_std_J, cd_hb_mean_h,cd_hb_std_h,cd_hb_mean_J,cd_hb_std_J, cd_mp_mean_h,cd_mp_std_h,cd_mp_mean_J,cd_mp_std_J") 
   # N_list = [80,160,320, 640, 960, 1280]
    #N_list_unit = [20,30,40,50,60,70,80,90,100]
     _list_unit = [50,100,200,500,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
     N_list = np.append(N_list_unit,N_list_unit) 
    N_list = np.append(N_list,N_list_unit) 
    s_time = time.time()
    for N in N_list:
        ml_list_h, ml_list_J = np.zeros(n_statis), np.zeros(n_statis)
        cd_hb_list_h, cd_hb_list_J= np.zeros(n_statis), np.zeros(n_statis)
        cd_mp_list_h, cd_mp_list_J= np.zeros(n_statis), np.zeros(n_statis)
        for m in range(n_statis):
            sample = gen_sample(N,h0,J0)
            global alpha, P0 
            alpha = 0.0
            P0 = empirical_dist(d,sample)
            sdata,ssdata = calc_data_hJ(sample)
            sdata0,ssdata0= calc_model_mle(h0,J0)
            #MLE
            solv_ml=fsolve(eq_of_ml,initial,args=(alpha,sdata0,ssdata0,sdata,ssdata)) - ans
               
            #CD1-HB
            solv_cd_hb=fsolve(calc_model_cd1_HB_LSE,initial,args=(h0,J0)) - ans
            #CD1-MP
            solv_cd_mp=fsolve(calc_model_cd1_MP_LSE,initial,args=(h0,J0)) - ans
            #print "solv_ml = \n", solv_ml


            #diff_ml_h,diff_ml_J = np.sqrt(sum((solv_ml[:d]-ans[:d])**2))/d, np.sqrt(sum((solv_ml[d:]-ans[d:])**2))/d
            #diff_cd_h,diff_cd_J = np.sqrt(sum((solv_cd[:d]-ans[:d])**2))/d, np.sqrt(sum((solv_cd[d:]-ans[d:])**2))/d
            diff_ml_h,diff_ml_J = sum(solv_ml[:d])/d, sum(solv_ml[d:])/d
            diff_hb_h,diff_hb_J = sum(solv_cd_hb[:d])/d, sum(solv_cd_hb[d:])/d
            diff_mp_h,diff_mp_J = sum(solv_cd_mp[:d])/d, sum(solv_cd_mp[d:])/d
            
            ml_list_h[m],ml_list_J[m] = diff_ml_h,diff_ml_J
            cd_hb_list_h[m],cd_hb_list_J[m] = diff_hb_h,diff_hb_J
            cd_mp_list_h[m],cd_mp_list_J[m] = diff_mp_h,diff_mp_J
        #ML
        ml_mean_h,ml_std_h = np.mean(ml_list_h),np.std(ml_list_h)
        ml_mean_J,ml_std_J = np.mean(ml_list_J),np.std(ml_list_J)
        #CD-HB
        cd_hb_mean_h,cd_hb_std_h = np.mean(cd_hb_list_h),np.std(cd_hb_list_h)
        cd_hb_mean_J,cd_hb_std_J = np.mean(cd_hb_list_J),np.std(cd_hb_list_J)
        #CD-MP
        cd_mp_mean_h,cd_mp_std_h = np.mean(cd_mp_list_h),np.std(cd_mp_list_h)
        cd_mp_mean_J,cd_mp_std_J = np.mean(cd_mp_list_J),np.std(cd_mp_list_J)
        #ml_vari[m],cd_vari = diff_ml,diff_cd
        #print(N, ml_mean_h,ml_std_h, ml_mean_J,ml_std_J, cd_mean_h,cd_std_h,cd_mean_J,cd_std_J) 
        print(N, ml_mean_h, ml_mean_J, cd_hb_mean_h,cd_hb_mean_J, cd_mp_mean_h,cd_mp_mean_J) 
        f.write( str(N)+"  "+str(ml_mean_h)+"  " +str(ml_std_h) + "  " + str(ml_mean_J)+ "  " + str(ml_std_J) +"  "+str(cd_hb_mean_h)+"  " +str(cd_hb_std_h) + "  " + str(cd_hb_mean_J)+ "  " + str(cd_hb_std_J) + "  "+str(cd_mp_mean_h)+"  " +str(cd_mp_std_h) + "  " + str(cd_mp_mean_J)+ "  " + str(cd_mp_std_J) +"\n")
    f_time = time.time()
    f.write("#computational time = "+ str(f_time-s_time))
    f.close()


