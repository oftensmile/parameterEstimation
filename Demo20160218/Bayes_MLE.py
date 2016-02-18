import numpy  as np 
from scipy  import linalg
import matplotlib.pyplot as plt
np.random.seed(0)
M,alpha,beta,sample=10,0.0005,11.1,10
def phi(x):
    p=np.zeros(M)
    for i in range(M):p[i]=x**i
    return p
train_in=np.linspace(0,2*np.pi,sample)
train_out=np.sin(train_in)+0.05*np.random.randn(sample)
#a=np.arange(M)
#check of parameter estimation
#train_out=np.zeros(sample)
#for i in range(M):train_out=train_out+a[i]*train_in**i
#train_out[5]=-2#abnormal value
for i in range(sample):
    if i==0:
        p_sample=phi(train_in[i])
    else:
        p_sample=np.vstack((p_sample,phi(train_in[i])))
Sinv=alpha*np.eye(M)+beta*np.matrix(p_sample).T*np.matrix(p_sample)
S=np.linalg.inv(Sinv)
phi_t=np.dot(p_sample.T,train_out.T)
Sphi_t=np.dot(S,phi_t.T)
#p_sample_mle=p_sample/(np.sqrt(sample))
phiTphi=np.dot(p_sample.T, p_sample)
#phiTphi_mle=phiTphi/sample
A=beta/sample*phiTphi+alpha*np.eye(M)
Ainv=np.linalg.inv(A)
#w_mle=np.matrix(np.dot(Ainv,phi_t/np.sqrt(sample))).T
w_mle=np.matrix(np.dot(Ainv,phi_t)).T
line_x=np.linspace(0,2*np.pi,500)
for t in range(len(line_x)):
    pofx=phi(line_x[t])
    if t==0:
        m=beta*np.dot(np.matrix(pofx),Sphi_t.T)
        ss=1.0/beta + np.dot(np.dot(pofx,S),pofx.T)
        m_mle=np.dot(np.matrix(pofx),w_mle)
    else:
        m=np.hstack((np.matrix(m),beta*np.dot(np.matrix(pofx),Sphi_t.T)))
        ss=np.hstack((ss,1.0/beta + np.dot(np.dot(pofx,S),pofx.T)))
        m_mle=np.hstack((m_mle,np.dot(np.matrix(pofx),w_mle)))
ss=np.array(ss)[0]
m=np.array(m)[0]
m_mle=np.array(m_mle)[0]
plt.plot(line_x,np.sin(line_x),'g-')
plt.plot(train_in,train_out,'bo')
pb,=plt.plot(line_x,m,'r-',label='pb')
pbu,=plt.plot(line_x,m+ss,'r--',label='pbu')
pbl,=plt.plot(line_x,m-ss,'r--',label='pbl')
pmle,=plt.plot(line_x,m_mle,'b-',label='pmle')
#plt.legend([pb,pbu,pbl,pmle],['pb','pbu','pbl','pmle'])
plt.legend([pb,pmle],['pb','pmle'])
print("w_mle",w_mle)
print("w_Bayes",Sphi_t)
plt.show()

    


#ss=1.0/beta + np.dot(phi(x),np.dot(S,phi(x)))

