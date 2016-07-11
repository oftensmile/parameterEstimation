import numpy as np
import scipy as sp 
import time
np.random.seed(0)

d=8
beta=1.5
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
        error=np.sum(np.abs(m-temp_m))
        print(error)
        if(error<eps):
            print(m)
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
        print(t,error)
        if(error<eps):
            break 
    return c

def gen_mcmc(x=[]):
    #Heat Bath
    for i in range(d):
        diff_E=0.0
        diff_E=beta*x[i]*(np.dot(J[i],x)+np.dot(J[j],x))
        r=1.0/(1+np.exp(diff_E)) 
        #r=np.exp(-diff_E) 
        R=np.random.uniform(0,1)
        if(R<=r):
            x[i]=x[i]*(-1)
    return x


if __name__ == '__main__':
    x=np.ones(d)
    for n in range(N_sample+N_remove):
        for t in range(t_interval):
            x=np.copy(gen_mcmc(x))
        if(n==N_remove):X_sample=np.copy(x)
        elif(n>N_remove):X_sample=np.vstack((X_sample,np.copy(x)))
    
    m_mcmc=np.zeros(d)
    len_sample=len(X_sample)
    for i in range(d):
        m_mcmc[i]=np.sum(X_sample.T[i])*(1.0/len_sample)
    print("#m_mcmc=",m_mcmc)


    m=np.ones(d)
    calc_m_MF_SCeq(100,0.00001,m)
    print("#m=",m)
    c=np.ones((d,d))
    for i in range(d):c[i][i]=0
    c=calc_c_MF_SCeq(100,0.00001,m)
    print("#c=",c)
    print("#J=",J)
