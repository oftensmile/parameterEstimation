import numpy as np
import scipy as scipy
from numpy.linalg import inv
import sys
import matplotlib.pyplot as plt
N=1000
epc_max=300
mu0, sig0=10.0,2.0
iota=0.001
data=mu0+sig0*np.random.randn(N)
#data/=np.sqrt(2*3.141516*sig0**2)

mu_gd, sig_gd=3.0,3.0   #sd*sd=sig
mu_nm, sig_nm=3.0,3.0
eps=0.001
x_sum=np.sum(data)
xx_sum=np.sum(data*data)

def mle():
    mu=x_sum/N
    var=(xx_sum+N*mu**2-2*x_sum*mu)/N
    return (mu,var)

def SG(mu, sig):
    for i in range(epc_max):
        mu+=iota*(-1.0/sig)*(N*mu-x_sum)
        sig+=iota*(-N/(2.0*sig)+1.0/(2.0*sig**2)*(xx_sum-2.0*mu*x_sum+N*mu**2) )
        del_mu=np.abs(mu0-mu)
        del_sig=np.abs(sig0*sig0-sig)
        print(del_mu,del_sig)
    return (mu,sig)

def Newton(mu,sig):
    h=np.ones((2,2))
    theta=[mu,sig]
    err,eps=10,0.1
    #for i in range(epc_max):
    while(err>=eps):
        h[0][1]=(N*mu-x_sum)/(sig**2)
        h[1][0]=h[0][1]
        h[0][0]=-N/sig
        #h[1][1]=N/(2.0*sig**2)-(xx_sum-2.0*mu*x_sum+N*mu**2)/sig**3
        h[1][1]=N/(2.0*sig**2)-(xx_sum-2.0*mu*x_sum+N*mu**2)/sig**3
        hinv=inv(h)
        #if(i==0 or i>epc_max-3):print("#H*Hinv=",np.dot(h,hinv))

        grad_mu=(-1.0/sig)*(N*mu-x_sum)
        grad_sig=(-N/(2.0*sig)+1.0/(2.0*sig**2)*(xx_sum-2.0*mu*x_sum+N*mu**2) )
        grad=np.array([grad_mu,grad_sig])
        delta=np.dot(hinv,grad)
        theta=theta-iota*delta
        err=np.sum(np.abs(delta))
        print(np.abs(theta[0]-mu0),np.abs(theta[1]-sig0))
    mu,sig=theta[0],theta[1]
    return (mu,sig)



if __name__ == '__main__':
    param=sys.argv
    p=int(param[1])
    print(param[1])
    if(p==0):
        print("#GD")
        mu_gd,sig_gd=SG(mu_gd,sig_gd)
        print("#mu_gd=",mu_gd, "sig_gd=",sig_gd)
    if(p==1):
        print("#Newton")
        mu_nm,sig_nm=Newton(mu_nm,sig_nm)
        print("#mu_nm=",mu_nm, "sig_nm=",sig_nm)
    
    mu_mle,var_mle=mle()
    print("#simple MLE")
    print("#=",mu_mle,var_mle)
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.hist(data, bins=80)
    fig.savefig("notitle.png")
    fig.show()
