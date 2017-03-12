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
    s=[]
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
        s.append(x)
    mean_xx /= float(N)
    mean_xandx /= float(N)
    #p1_emp = 0.25*(1+mean_xx+mean_xandx)
    #p2_emp = 0.25*(1+mean_xx-mean_xandx)
    return ( mean_xx,mean_xandx,np.copy(s) )

def solv_hJ(xx,xandx):
    h = 0.25*np.log( (1+xx+xandx) / (1+xx-xandx) )
    J = 0.25*np.log( (1+xx+xandx)*(1+xx-xandx) / (1-xx)**2 )
    return (h,J)

def mleq_of_hJ(h,J,xx,xandx):
    ch,sh,eJ = np.cosh(2*h), np.sinh(2*h), np.exp(-2*J)
    mleq1 = xx- (ch-eJ) / (ch+eJ)
    mleq2 = xandx - 2*sh / (ch+eJ) 
    return (mleq1,mleq2)

def loss_func_ml(h,J,xx,xax):
    return J*xx+h*xax-np.log(2*np.exp(J)*np.cosh(2*h)+2*np.exp(-J))

def loss_func_cd(h,J,a,b,xx,xax):
    q1,q4 = 0.5*(xx+xax/2.0), 0.5*(xx-xax/2.0)
    xx1, xax1 = 2*( (1-a+(a-b)/2.0)*q1+(b-1+(a-b)/2.0)*q4+(b-a)/2.0), -(a+b-2)*(q1-q4)-(a-b) 
    return J*(xx-xx1)+h*(xax-xax1)

def element_Tmat(h,J):
    a, b = 1.0/(1+np.exp(2.0*(J+h))), 1.0/(1+np.exp(2.0*(J-h))) 
    return (a,b) 

def loss_func_cd_master(h,J,s=[]):
    n=len(s)
    loss_func=0.0
    for x in s:
        xx=x[0]*x[1]
        xax=x[0]+x[1]
        loss_func += np.log(np.exp(-2.0*h*xax)+1.0)/float(n)
        +2.0*np.log(np.exp(-2.0*J*xx-h*xax)+1.0)/float(n)
    return loss_func

if __name__ == '__main__':
    h0,J0 =0.0, 0.8
    N_list = [10,40]
    M = 100
    n_mesh = 100
    h_min,h_max,J_min,J_max = 0.05,1.55,0.05,1.55
    h_list = np.linspace(0.05,1.55,100)
    J_list = np.linspace(0.05,1.55,100)
    for N in N_list:
        fname="plot-sav-"+str(M)+"-sample-"+str(N)+"-J0-"+str(J0)+"-h0-"+str(h0)+"-0310.dat"
        f=open(fname,"w")
        f.write("#N, h, J, loss_func_ml, loss_func_cd \n")
        m = 0
        mean_ml = np.zeros((n_mesh,n_mesh))
        #mean_cd = np.zeros((n_mesh,n_mesh))
        mean_cd_master = np.zeros((n_mesh,n_mesh))
        while(m<M):
            m+=1
            mean=mean_stat(h0,J0,N)
            xx,xax,s = mean[0],mean[1],mean[2]
            if((1+mean[0]+mean[1])!=0 and (1+mean[0]-mean[1])!=0):
                for i in range(n_mesh):
                    h = h_list[i]
                    for j in range(n_mesh):
                        J = J_list[j]
                        a,b = element_Tmat(h,J)
                        mean_ml[i][j] += loss_func_ml(h,J,xx,xax) /float(M)
                        #mean_cd[i][j] += loss_func_cd(h,J,a,b,xx,xax) /float(M)
                        mean_cd_master[i][j] += loss_func_cd_master(h,J,s) /float(M)
                        if(m==M):
                            #print h,J, mean_ml[i][j], mean_cd[i][j], mean_cd_master[i][j]
                            print h,J, mean_ml[i][j], mean_cd_master[i][j]
                            f.write(str(h)+" "+str(J)+"  "+str(mean_ml[i][j])+"  "+str(-mean_cd_master[i][j])+"\n")
            else:
                m -=1
        f.close()
