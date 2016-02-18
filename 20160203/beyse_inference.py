import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
M,alpha,beta=9, 0.005,11.1

#M-polynomial approximation
def y(x,w=[]):
    ret=wlist[0]
    for i in range(1,M+1):
        ret+=wlist[i]*(x**i)
    return ret

def phi(x):
    data=[]
    for i in range(0,M+1):
        npappend(data,x**i)
    ret=np.matrix(data).reshape((M+1,1))
    return ret
def mean(x,xlist,tlist,S):
    sums=np.matrix(np.zeros((M+1,M+1)))
    for n in range(xlist[n]):
        sums+=phi(xlist[n])*tlist[n]
    ret =beta*phih(x).transpose()*S*sums
    return ret
def variance(x,xlist,S):
    ret=1.0/beta +phi(x).transpose()*S*phi(x)

def main():
    xlist=np.linspace(0,1,10)
    tlist=np.sin(2*np.pi*xlist)+np.random.normal(0,0.2,xlist.size)

    sums=np.matrix(np.zeros((M+1,M+1)))
    for n in range(len(xlist)):
        sums+=phi(xlist[n]*phi(xlist[n]).transpose())
    I=np.matrix(np.identity(M+1))
    S_inv=alpha*I+beta*sums
    S=S_inv.getI()
    xs=np.linspace(0,1,500)
    ideal=np.sin(2*np.pi*xs)
    means,uppers,lowers=[],[],[]
    for x in xs:
        m=mean(x,xlist,tlist,S)[0,0]
        s=np.sqrt(variance(x,xlist,S)[0,0])
        u,=m+s,m-s
        means.append(m)
        uppers.append(u)
        lowers.append(l)
    plot(xlist,tlist,'bo')
    plot(xs,ideal,'g-')
    plot(xs,means,'r-')
    plot(xs,uppers,'r--')
    plot(xs,lowers,'r--')
    xlim(0.0,1.0)
    ylim(-1.5,1.5)
    show()

if __name__ =="__main__":
    main()
        
