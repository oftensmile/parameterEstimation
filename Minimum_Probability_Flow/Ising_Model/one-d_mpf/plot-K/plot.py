from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
data=np.loadtxt('./data_koft.dat',comments="#",delimiter=" ")
x=data.T[0]
y=data.T[1]
z=data.T[2]
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter3D(x,y,z)
