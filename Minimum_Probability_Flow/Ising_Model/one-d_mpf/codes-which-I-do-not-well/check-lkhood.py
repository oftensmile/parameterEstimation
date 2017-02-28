import numpy as np
from scipy import linalg 
import matplotlib.pyplot as plt
np.random.seed(0)
n_sample=500
mu,sig2=0.0,1.0
nor=np.random.normal(mu,sig2,n_sample)
nor2=np.random.normal(mu,sig2,n_sample*2)
nor4=np.random.normal(mu,sig2,n_sample*4)
nor8=np.random.normal(mu,sig2,n_sample*8)
nor16=np.random.normal(mu,sig2,n_sample*16)
plt.hist(nor,bins=30,alpha=0.5)
plt.hist(nor2,bins=30,alpha=0.5)
plt.hist(nor4,bins=30,alpha=0.5)
plt.hist(nor8,bins=30,alpha=0.5)
plt.hist(nor16,bins=30,alpha=0.5)
fig1=plt.figure()
ax1.plot()
ax1=fig1.add_subplot(111)
plt.hist(nor,bins=30,alpha=0.5)
plt.hist(nor2,bins=30,alpha=0.5)
plt.hist(nor4,bins=30,alpha=0.5)
plt.hist(nor8,bins=30,alpha=0.5)
plt.hist(nor16,bins=30,alpha=0.5)
fig2=plt.figure()
ax2=fig2.add_subplot(111)
plt.hist(nor,bins=30,alpha=0.5)
plt.show()
