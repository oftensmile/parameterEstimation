# plotting generated data files
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

data1 = np.loadtxt('output_gibbus_sample5.txt',comments='#')
data2 = np.loadtxt('output_gibbus_sample5_1.txt',comments='#')
data3 = np.loadtxt('output_gibbus_sample5_2.txt',comments='#')
data4 = np.loadtxt('output_gibbus_sample5_3.txt',comments='#')


plt.plot(data1, label='ypc = 0.1')
plt.plot(data2, label='ypc = 1.0 / log(2 + epoch)')
plt.plot(data4, label='ypc = 1.5 / log(2 + epoch)')
plt.plot(data3, label='ypc = 2.0 / log(2 + epoch)')
plt.legend()
plt.ylabel('Error function', fontsize='20')
plt.xlabel('epoch', fontsize='20')

plt.show()

