import numpy as np
import numpy as np
from scipy import linalg
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
import numpy.matlib
np.set_printoptions(precision=2)
np.random.seed(0)
N,hh,m=100,1.0**2,2#sample,#width,#enbed
a,b=3.0*np.pi*np.random.rand(N),3.0*np.pi*np.random.rand(N)
x=np.array([np.append(a*np.cos(a),0.6*b*np.cos(b)),np.append(20*np.random.rand(N),20*np.random.rand(N)),np.append(a*np.sin(a),0.6*b*np.sin(b))])
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter3D(np.ravel(x[0]),np.ravel(x[1]),np.ravel(x[2]),c=np.append(a,b))
plt.show()
xx=np.dot(x.T,x)# =(60,60)
x2=np.matrix(np.diagonal(xx))
x2=np.matlib.repmat(x2,len(x2),1)+np.matlib.repmat(x2.T,1,len(x2))
K=np.exp(-(x2-2*xx)/hh)
H=np.eye(2*N)-np.ones((2*N,2*N))/(2*N)
#make mean of K is 0
K=np.dot(H,np.dot(K,H))
U,s,Vh=linalg.svd(K)
lamda=np.zeros((m,m))
for i in range(m):lamda[i][i]=s[i] 
A=np.array([U[0],U[1]])
print(np.shape(A))
B=np.dot(np.dot(np.sqrt(lamda),A),K)
print(np.shape(B))
plt.plot(B[0],B[1],'o')
plt.show()
