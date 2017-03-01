#! /usr/bin/env python
#-*-coding:utf-8-*-
import random, math
import numpy as np

def partition(h,J):
    return 2 * ( np.exp(J)*np.cosh(2*h)+np.exp(-J) ) 

def mean_stat(h,J,N):
    Z = partition(h,J)
    p1,p2 = np.exp(J+2*h)/Z, np.exp(J-2*h)/Z 
    mean_xx,mean_xandx=0.0,0.0
    for i in range(N):
        r = random.uniform(0.0,1.0)
        if(r<p1):
            x=[1,1]
        elif(p1<=r and r<p1+p2):
            x=[-1,-1]
        elif(p1+p2<=r and r<1):
            x=[-1,1]
        mean_xx += x[0] * x[1]
        mean_xandx += x[0] + x[1]
    mean_xx /= float(N)
    mean_xandx /= float(N)
    #p1_emp = 0.25*(1+mean_xx+mean_xandx)
    #p2_emp = 0.25*(1+mean_xx-mean_xandx)
    #print "    p=", p1, p2
    #print "p_emp=", p1_emp, p2_emp
    return ( mean_xx,mean_xandx )

def solv_hJ(xx,xandx):
    h = 0.25*np.log( (1+xx+xandx) / (1+xx-xandx) )
    J = 0.25*np.log( (1+xx+xandx)*(1+xx-xandx) / (1-xx)**2 )
    return (h,J)

def mleq_of_hJ(h,J,xx,xandx):
    ch,sh,eJ = np.cosh(2*h), np.sinh(2*h), np.exp(-2*J)
    mleq1 = xx- (ch-eJ) / (ch+eJ)
    mleq2 = xandx - 2*sh / (ch+eJ) 
    return (mleq1,mleq2)

if __name__ == '__main__':
    h0,J0 =0.2, 0.1
    N_list = [40,80,120,160,240,320,480,640,960,1280,1920,2560,3840,5120,7680]
    M_list = [10,100,1000,10000,100000]
    for M in M_list:
        fname="stav-"+str(M)+"-J0-"+str(J0)+"-h0-"+str(h0)+".dat"
        f=open(fname,"w")
        f.write("#N, bias_h, bias_J, b_std_h/sqrt(M), b_std_J/sqrt(M) \n")
        for N in N_list:
            bh_list = np.zeros(M)
            bJ_list = np.zeros(M)
            #for m in range(M): 
            m = 0
            while(m<M):
                mean=mean_stat(h0,J0,N)
                if((1+mean[0]+mean[1])!=0 and (1+mean[0]-mean[1])!=0):
                    h,J = solv_hJ(mean[0],mean[1]) 
                    check = mleq_of_hJ(h,J,mean[0],mean[1])
                    bh_list[m], bJ_list[m] = h-h0, J-J0 
                else:
                    m -=1
                m+=1
            bias_h, b_std_h = np.mean(bh_list), np.std(bh_list)
            bias_J, b_std_J = np.mean(bJ_list), np.std(bJ_list)
            #print M, N, bias_h, bias_J, b_std_h, b_std_J 
            f.write(str(N) + "  " + str(abs(bias_h)) + "  "  + str(b_std_h/np.sqrt(M)) 
                    + "  " + str(abs(bias_J)) + "  "  + str(b_std_J/np.sqrt(M))
                    +"\n" )
        f.close()    
