import numpy as np
import numpy as np
from scipy import linalg
from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
import numpy.matlib
np.set_printoptions(precision=2)
np.random.seed(0)
N,hh,m=1000,35.0,2#sample,#width,#enbed
a=3.0*np.pi*np.random.rand(N)
x=np.array([a*np.cos(a),20*np.random.rand(N),a*np.sin(a)])
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter3D(np.ravel(x[0]),np.ravel(x[1]),np.ravel(x[2]),c=a)
plt.show()
H=np.eye(N)-np.ones((N,N))/N
x=np.dot(x,H)
xx=np.dot(x.T,x)# =(60,60)

la,v=linalg.eig(xx)
idx=la.argsort()[::-1]
l1,l2=la[idx[0]],la[idx[1]]
diag=np.diag([l1,l2])
v1,v2=v[:,idx[0]],v[:,idx[1]]
A=np.vstack((v1,v2))
B=np.dot(np.sqrt(diag),np.dot(A,xx))
plt.scatter(B[0],B[1],c=a)
plt.show()
