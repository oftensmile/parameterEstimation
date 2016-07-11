import numpy as np
import scipy as sp 
import time
import matplotlib.pyplot as plt
from numpy.linalg import inv
np.random.seed(0)

d=8
beta=1.0
J=np.random.choice([-1,1],(d,d))
#J=np.random.choice([0,1],(d,d))
N_sample,N_remove=300,20
t_interval=20
for i in range(d):
    for j in range(i+1,d):
        J[j][i]=J[i][j]
for i in range(d):J[i][i]=0
#h=np.random.randn(0,0.5,d)
def calc_m_MF_SCeq(epc_max,eps,m=[]):
    len_m=len(m)
    for t in range(epc_max):
        temp_m=np.copy(m)
        for i in range(len_m):
            m[i]=np.tanh(np.dot(J[i],temp_m))
            """
            #Onsager Reacton Term (TAP)
            temp_mi=0.0
            for j in range(d):
                temp_mi+=J[i][j]*(temp_m[j]-J[i][j]*(1-temp_m[j]**2)*temp_m[i])
            m[i]=np.tanh(temp_mi)
            """
        error=np.sum(np.abs(m-temp_m))
        #print(t,error)
        if(error<eps):
            break
    return m

def calc_c_MF_SCeq(epc_max,eps,m=[]):
    c=np.ones((d,d))
    for i in range(d):c[i][i]=0
    for t in range(epc_max):
        temp_c=np.copy(c)
        for i in range(d):
            for j in range(i+1,d):
                if(i==j):
                    c[i][j]=(1.0-m[i]**2) * (1+np.dot(J[i],temp_c.T[j]))
                    c[j][i]=(1.0-m[j]**2) * (1+np.dot(J[j],temp_c.T[i]))
                else:
                    c[i][j]=(1.0-m[i]**2) * np.dot(J[i],temp_c.T[j])
                    c[j][i]=(1.0-m[j]**2) * np.dot(J[j],temp_c.T[i])
        error=np.sum(np.sum(np.abs(c-temp_c)))
        #print(t,error)
        if(error<eps):
            break 
    return c

def calc_c_MF_Bethe_SCeq(m=[]):
    c=np.zeros((d,d))
    t=np.tanh(J)
    print("t[1][2]-thanh(J[1][2])=",np.tanh(J[1][2])-t[1][2])
    for i in range(d):
        for j in range(i+1,d):
            c[i][j]=1.0/(2*t[i][j])*(1+t[i][j]**2-np.sqrt((1.0-t[i][j]**2)**2-4.0*t[i][j]*(m[i]-t[i][j]*m[j])*(m[j]-t[i][j]*m[i])))
            c[j][i]=c[i][j]
    return c



def gen_mcmc(x=[]):
    #Heat Bath
    for i in range(d):
        diff_E=0.0
        diff_E=beta*x[i]*(np.dot(J[i],x)+np.dot(J.T[i],x))
        r=1.0/(1+np.exp(diff_E)) 
        #r=np.exp(-diff_E) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x

def calc_cinv_MF_SCeq(m=[]):
    cinv=-np.copy(J)
    for i in range(d):
        c[i][i]+=1.0/(1-m[i]**2)
    return cinv


if __name__ == '__main__':
    x=np.ones(d)
    for n in range(N_sample+N_remove):
        for t in range(t_interval):
            x=np.copy(gen_mcmc(x))
        if(n==N_remove):X_sample=np.copy(x)
        elif(n>N_remove):X_sample=np.vstack((X_sample,np.copy(x)))
    #calc m
    len_sample=len(X_sample)
    XnXn=np.zeros((d,d))
    for n in range(len_sample):
        xn=np.matrix(np.copy(X_sample[n]))
        XnXn=XnXn + (1.0/len_sample) * np.tensordot(xn,xn,axes=([0],[0]))
    for i in range(d):XnXn[i][i]=0.0
    m_mcmc=np.zeros(d)
    for i in range(d):
        m_mcmc[i]=np.sum(X_sample.T[i])*(1.0/len_sample)
    m_mcmc_mat=np.matrix(m_mcmc)
    print("#m_mcmc=",m_mcmc)
    mmt=np.tensordot(m_mcmc_mat,m_mcmc_mat,axes=([0],[0]))
    for i in range(d):mmt[i][i]=0.0
    c_mcmc=XnXn-mmt
    print("#mcmc_c=",c_mcmc)
    #calc c
    m=np.ones(d)
    m=calc_m_MF_SCeq(100,0.00001,m)
    print("#m=",m)
    c=np.ones((d,d))
    for i in range(d):c[i][i]=0
    #c=calc_c_MF_SCeq(100,0.00001,m)
    #print("#c=",c)
    #Bethe
    #c=calc_c_MF_Bethe_SCeq(m)
    #print("#c_bethe=",c)
    
    #linear Response
    cinv=np.ones((d,d))
    cinv=calc_cinv_MF_SCeq(m)
    cinv=calc_cinv_MF_SCeq(m)
    cinv_inv=inv(cinv)
    print("#c_LR=",cinv_inv)

    plt.figure()
    plt.subplot(131)
    plt.imshow(c_mcmc)
    plt.colorbar()
    plt.title("mcmc")
    plt.subplot(132)
    #plt.imshow(c)
    plt.imshow(cinv_inv)
    plt.colorbar()
    #plt.title("MF+Bethe")
    plt.title("Linear Response")
    plt.subplot(133)
    plt.imshow(c_mcmc-cinv_inv)
    plt.colorbar()
    plt.title("mcmc-approximaition")
    plt.show()

