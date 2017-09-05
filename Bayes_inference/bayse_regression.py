import numpy  as np 
from scipy  import linalg
import matplotlib.pyplot as plt
np.random.seed(0)
M,alpha,beta,sample=6,0.0005,11.1,100
def phi(x):
    p=np.zeros(M)
    for i in range(M):p[i]=x**i
    return p

train_in=np.linspace(0,2*np.pi,sample)
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
        ss=np.hstack((ss,1.0/beta + np.dot(np.dot(pofx,S),pofx.T)))
#lx=np.matrix(line_x)[0].T
m=np.array(m)[0]
ss=np.array(ss)[0]
plt.plot(line_x,np.sin(line_x),'g-')
plt.plot(train_in,train_out,'bo')
plt.plot(line_x,m,'r-')
plt.plot(line_x,m+ss,'r--')
plt.plot(line_x,m-ss,'r--')
plt.title('sin curve with bayes estimat')
print("parameter=",Sphi_t)
plt.show()

