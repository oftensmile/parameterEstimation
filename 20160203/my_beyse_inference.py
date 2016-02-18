import numpy  as np 
from scipy  import linalg
import matplotlib.pyplot as plt
np.random.seed(0)
M,alpha,beta,sample=4,0.0005,11.1,10
def phi(x):
    p=np.zeros(M)
    for i in range(M):p[i]=x**i
    return p#np.matrix(p)

train_in=np.linspace(0,2*np.pi,sample)
#train_out=np.matrix(np.sin(train_in)+np.random.randn(sample))
train_out=np.sin(train_in)+0.1*np.random.randn(sample)
for i in range(sample):
    if i==0:
        p_sample=phi(train_in[i])
    else:
        p_sample=np.vstack((p_sample,phi(train_in[i])))
Sinv=alpha*np.eye(M)+beta*np.matrix(p_sample).T*np.matrix(p_sample)
S=np.linalg.inv(Sinv)
phi_t=np.dot(p_sample.T,train_out.T)
Sphi_t=np.dot(S,phi_t.T)

line_x=np.linspace(0,2*np.pi,500)
for t in range(len(line_x)):
    pofx=phi(line_x[t])
    if t==0:
        m=beta*np.dot(np.matrix(pofx),Sphi_t.T)
        ss=1.0/beta + np.dot(np.dot(pofx,S),pofx.T)
    else:
        m=np.hstack((np.matrix(m),beta*np.dot(np.matrix(pofx),Sphi_t.T)))
lx=np.matrix(line_x)[0].T
plt.plot(line_x,np.sin(line_x),'g-')
plt.plot(train_in,np.array(train_out),'bo')
plt.plot(lx,np.matrix(m)[0],'r-')
plt.plot(lx,np.matrix(m+ss)[0],'r--')
plt.plot(lx,np.matrix(m-ss)[0],'r--')
print(np.max(m),np.min(m),len(m))
plt.show()

    


#ss=1.0/beta + np.dot(phi(x),np.dot(S,phi(x)))
