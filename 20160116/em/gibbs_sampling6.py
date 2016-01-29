import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
np.random.seed(0)

n = 16  # number of Ising variables
T = 500 # number of epoc for theta
N = 500 # number of smples, comes form true Prob
N_est = 20 # number of smples, comes form estimated Prob
ypc = 1 # descent ratio
#theta = np.tensordot(theta[:, np.newaxis], theta[: , np.newaxis], axes = ([1],[1]))
#np.fill_diagonal(theta, 0)
theta_est =  np.random.rand(n,n)    # create same size of matrix
np.fill_diagonal(theta_est, 0)
theta_tr = np.transpose(theta_est)
theta_est = theta_est + theta_tr
np.fill_diagonal(theta_est, 0)
theta=[[1 if (j+i+2)%n==0 or (j+i)%n==0  else 0 for i in range(n)] for j in range(n)]
theta = np.array(theta)

# set del theta mat
del_l_del_theta = np.empty_like(theta) # sum of x_i* x_j for each sample
del_l_del_theta_est = np.empty_like(theta) # sum of x_i* x_j for each sample
#X = np.ones(n)
#X_est = np.ones(n)

X =  2 * np.array(np.random.random_integers(0,1,n) - 0.5)
X_est =  2 * np.array(np.random.random_integers(0,1,n) - 0.5)
def gen_a_gibbus_sample(t_wait_sample,X = [], theta=[[]]):
    for i in range(t_wait_sample * n):# 20 = 収束するまでの更新回数
        cord = i  % n 
        valu =   ( np.tensordot(X, theta[:,cord], axes = ([0],[0]) ) -X[cord] * theta[cord][cord] ) 
        r = np.exp( -valu ) / ( np.exp( -valu ) + np.exp( valu ) )
        R = np.random.uniform(0,1)
        if (R <= r):
            X[cord] =  1
        else:
            X[cord] = -1
    return X

def get_del_l_del_theta_mat(N, X = [] , theta = [[]]):
    del_l_del_theta = np.zeros((n,n), dtype=np.float)
    for k in range(N):
        X = gen_a_gibbus_sample(10, X , theta)
        del_l_del_theta = del_l_del_theta + np.tensordot(X[:,np.newaxis],X[:,np.newaxis], axes = ([1],[1]))
    return del_l_del_theta / N
############################ MAIN ###############################
del_l_del_theta = get_del_l_del_theta_mat(N, X,theta)
data = np.zeros(T)
for t in range(T):
    ypc = 2.0 /np.log(t + 2)
    # update theta using Graduant Descent
    #theta_est = theta_est -  ypc * ( - del_l_del_theta_est )
    # sampling using t-th theta
    del_l_del_theta_est = get_del_l_del_theta_mat(N_est, X_est, theta_est)
    theta_est = theta_est -  ypc * ( del_l_del_theta - del_l_del_theta_est )
    data[t] = np.absolute( theta - theta_est ).sum()
    print( data[t] )

plt.subplot(121)
plt.imshow(theta_est,interpolation="none")
plt.subplot(122)
plt.plot(data)
plt.show()





np.set_printoptions(precision=4)
print("theta_est = \n", theta_est)

