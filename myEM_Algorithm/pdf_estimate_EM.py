##################################################################
#   This programming is intend to estimate pdf using EM Algorithm
#################################################################
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
np.random.seed(0)
def Gibs_Sampling(d,wait_t, x=[], theta=[[]]):
    for i in range(wait_t*d ):
        index = i % d
        #valu = (np.tensordot(x,theta[:,index], axes=([0],[0]) )- x[index]* theta[index][index])
        valu = np.dot(x,theta[:,index]) - x[index] * theta[index][index]
        r = np.exp(-valu) / (np.exp(-valu) + np.exp(valu))
        R = np.random.uniform(0,1)
        if (R<=r):
            x[index] = 1
        else:
            x[index] = -1
    return x

d = 8 #d:number of sipn(dimension of x),
N_true = 800 # number of sample ~ P(theta_true)
M = 800 # number of sample ~ P(theta)
T = 10 # the number of Q's update 
mcmc_wait_init, mcmc_wait= 500, 50

error = np.zeros(T)
theta_true =[[0,1,0,0,0,0,0,1],
            [1,0,1,0,0,0,0,0],
            [0,1,0,1,0,0,0,0],
            [0,0,1,0,1,0,0,0],
            [0,0,0,1,0,1,0,0],
            [0,0,0,0,1,0,1,0],
            [0,0,0,0,0,1,0,1],
            [1,0,0,0,0,0,1,0]]
theta_true = np.array(theta_true)
theta= np.random.rand(d,d)
theta = theta + theta.T
np.fill_diagonal(theta,0)
#   Learning sample set
x_true =  2 * np.array(np.random.random_integers(0,1,d) - 0.5)
for i in range(N_true):
    if (i==0):
        Gibs_Sampling(d, mcmc_wait_init, x_true, theta_true)
        X_true = x_true
    if (i!=0):
        Gibs_Sampling(d, mcmc_wait, x_true, theta_true)
        X_true = np.vstack((X_true, x_true))

# GEM Algorithm 
Q_old = -10000
for t in range(T):
    for i in range(d):
        for j in range(i+1,d):
            # create M-th samples, which obey ~ exp(-x.T * theta * x)
            x = 2 * np.array(np.random.random_integers(0,1,d) - 0.5)  
            for m in range(M):
                if(m==0):
                    Gibs_Sampling(d, mcmc_wait_init, x, theta)
                    X = x
                if(m!=0):
                    Gibs_Sampling(d, mcmc_wait, x, theta)
                    X = np.vstack((X, x))

            delta_theta = np.random.rand(1) -0.5 # = theta_prop - theta
# a := sum_m{ exp(-xmi * del_theta * xmj) }, xm ~ exp(-x.T * theta * x)
# b := sum_n{ exp(-xn.T * theta * xn) } , xn ~ exp(-x.T * theta_true * x)
# c := sum_n{ xni * xnj * exp(-xn.T * theta * xn) }
            a = delta_theta * X[:,i] * X[:,j]
            a = np.sum( np.exp( -a ) )
            a /= M
            b = np.zeros(N_true)
            for n in range(N_true):# maybe this part can be more simple
                b[n] = np.dot( X_true[n,:].T , np.dot(theta, X_true[n,:]) )
            b = np.exp( - b )
            c = delta_theta * X_true[:,i] * X_true[:,j]
            c = c * b
            #E-step
            Q_new = - np.sum(c) - np.log(a) * np.sum(b)
            #GM-step
            if (Q_new >= Q_old or (t==0 and i==0 and j==1)):
                theta[i,j] += delta_theta
                theta[j,i] += delta_theta
                Q_old = Q_new
            print(Q_old)
    error[t] = np.sum(abs(theta - theta_true))
    #print(error[t])
#   just output of estimated matrix
np.set_printoptions(precision=4)
with open("matrix.txt","w") as file:
    file.write("theta_true = \n")
    file.write(str(theta_true))
    file.write("\ntheta = \n")
    file.write(str(theta))
plt.plot(error)
plt.show()
