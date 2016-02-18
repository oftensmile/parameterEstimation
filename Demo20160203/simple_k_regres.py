import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from scipy import linalg
np.random.seed(0)
h,lam=1**2,0.1

def kernel_regres(x,target):
    x2=x**2
    x2=np.matlib.repmat(np.matrix(x2).T, 1,len(x))+np.matlib.repmat(np.matrix(x2),len(x),1)
    xx=np.dot(np.matrix(x).T,np.matrix(x))
    K=np.exp(-(x2-2.0*xx)/h)
    print("shape(K)",np.shape(K))
    Kinv=np.linalg.inv(K+lam*np.diag(np.ones(len(x))))#+lam*np.diag(np.ones(len(x)))
    #plt.subplot(131)
    #plt.imshow(K)
    #plt.title("K")
    #plt.subplot(132)
    #plt.imshow(Kinv)
    #plt.title("Kinv")
    #plt.subplot(133)
    #plt.imshow(np.dot(K,Kinv))
    #plt.title("K*Kinv")
    #plt.show()
    return np.dot(Kinv,target)

def k_of_x(x,sample):
    kofx=np.zeros(len(sample))
    for i in range(len(sample)):
        kofx[i]=np.exp(-(sample[i]**2+x**2-2*x*sample[i])/h)
    return kofx

Nsamp,Nfit=10,100
x=np.linspace(-3,3,Nsamp)
target_true=np.sin(x)
target=target_true+0.1*np.random.randn(len(x))
X=np.linspace(-3,3,Nfit)
for l in range(len(X)):
    if l==0:
        k=k_of_x(X[l],x)
    else:
        k=np.vstack((k,k_of_x(X[l],x)))

alpha=kernel_regres(x,target)
Y=np.dot(np.matrix(k),np.matrix(alpha).T)

p1,=plt.plot(x,target,'o')
p2,=plt.plot(X,np.sin(X))
print("len(X), len(Y)", len(X),len(Y))
p3,=plt.plot(X,Y,'--')
plt.legend([p1,p2,p3],['sample','target','regression'])
plt.show()

