import numpy as np
import scipy as scipy
from numpy.linalg import inv
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


N=1000
epc_max=100
mu=[1.0,1.0]
cov=[[1.0,1.0],[1.0,2.0]]
x1,x2=np.random.multivariate_normal(mu,cov,N).T
mean=[np.sum(x1)/N,np.sum(x2)/N]
def GD(mu=[]):
    v=[3.0,-3.0]
    for i in range(epc_max):
        grad_l=
        v=v-0.1*np.dot(cov,)

#plt.plot(x1,x2,'x')
#plt.axis('equal')
#plt.show()










fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
hist, xedges,yedges=np.histogram2d(x1,x2,bins=20)
elements = (len(xedges) - 1) * (len(yedges) - 1)
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25)
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros(elements)
dx = 0.5 * np.ones_like(zpos)
dy = dx.copy()
dz = hist.flatten()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
plt.show()
