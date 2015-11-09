# plotting generated data files
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

data1 = np.loadtxt('../data_change_sample_number/sample3000.dat',comments='#')
data2 = np.loadtxt('sample3000_rep5.dat',comments='#')
data3 = np.loadtxt('sample3000_rep10.dat',comments='#')


plt.plot(data1, label='num_rep=3')
plt.plot(data2, label='num_rep=5')
plt.plot(data3, label='num_rep=10')
plt.legend()
plt.ylabel('Error function', fontsize='20')
plt.xlabel('theta update-steps', fontsize='20')

plt.show()

