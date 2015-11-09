# plotting generated data files
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

data1_2 = np.loadtxt('sample2000.dat',comments='#')
#data1_3 = np.loadtxt('sample3000.dat',comments='#')
data1_4 = np.loadtxt('sample4000.dat',comments='#')
#data1_5 = np.loadtxt('sample5000.dat',comments='#')
#data2 = np.loadtxt('sample10000.dat',comments='#')
data3 = np.loadtxt('sample15000.dat',comments='#')
data4 = np.loadtxt('sample20000.dat',comments='#')


plt.plot(data1_2, label='num_sample=2000')
#plt.plot(data1_3, label='num_sample=3000')
plt.plot(data1_4, label='num_sample=4000')
#plt.plot(data1_5, label='num_sample=5000')
#plt.plot(data2, label='num_sample=10000')
plt.plot(data3, label='num_sample=15000')
plt.plot(data4, label='num_sample=20000')
plt.legend()
plt.ylabel('Error function', fontsize='20')
plt.xlabel('theta update-steps', fontsize='20')

plt.show()

