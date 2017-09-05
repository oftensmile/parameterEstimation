# plot of dta files
import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt


file1 = np.loadtxt('N_est_01.dat', comments='#')
file2 = np.loadtxt('N_est_02.dat', comments='#')
file3 = np.loadtxt('N_est_03.dat', comments='#')
file4 = np.loadtxt('N_est_04.dat', comments='#')
file5 = np.loadtxt('N_est_05.dat', comments='#')
file6 = np.loadtxt('N_est_10.dat', comments='#')
file7 = np.loadtxt('N_est_20.dat', comments='#')
file8 = np.loadtxt('N_est_40.dat', comments='#')
file9 = np.loadtxt('N_est_80.dat', comments='#')
#file10 = np.loadtxt('N_est_100.dat', comments='#')

plt.plot(file1, label  = 'N=01')
plt.plot(file2, label  = 'N=02')
plt.plot(file3, label  = 'N=03')
plt.plot(file4, label  = 'N=04')
plt.plot(file5, label  = 'N=05')
plt.plot(file6, label  = 'N=10')
plt.plot(file7, label  = 'N=20')
plt.plot(file8, label  = 'N=40')
plt.plot(file9, label  = 'N=80')
#plt.plot(file10, label  = 'N=100')
plt.legend()

plt.ylabel('Error function', fontsize='20')
plt.xlabel('theta step', fontsize='20')
plt.show()
