import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
np.random.seed(0)

n = 32  # number of Ising variables
T = 5000 # number of epoc for theta
N = 5000 # number of smples, comes form true Prob
N_est = 40 # number of smples, comes form estimated Prob
#theta = np.tensordot(theta[:, np.newaxis], theta[: , np.newaxis], axes = ([1],[1]))
#np.fill_diagonal(theta, 0)
theta_est =  0.1*np.random.rand(n,n)    # create same size of matrix
np.fill_diagonal(theta_est, 0)
theta_tr = np.transpose(theta_est)
theta_est = theta_est + theta_tr
np.fill_diagonal(theta_est, 0)
theta=[[1 if (j+i+2)%n==0 or (j+i)%n==0  else 0 for i in range(n)] for j in range(n)]
theta = np.array(theta)

# set del theta mat
del_l_del_theta = np.empty_like(theta) # sum of x_i* x_j for each sample
del_l_del_theta_est = np.empty_like(theta) # sum of x_i* x_j for each sample
X =  2 * np.array(np.random.random_integers(0,1,n) - 0.5)
X_est =  2 * np.array(np.random.random_integers(0,1,n) - 0.5)
def gen_a_gibbus_sample(t_wait_sample,X = [], theta=[[]]):
    for i in range(t_wait_sample * n):
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
    del_l_del_theta_est = get_del_l_del_theta_mat(N_est, X_est, theta_est)
    theta_est = theta_est -  ypc * ( - del_l_del_theta_est )
    # sampling using t-th theta
    data[t] = np.absolute( theta - theta_est ).sum()/(n**2)
    print( data[t] )

plt.subplot(131)
plt.imshow(theta)
plt.title("theta true")
plt.subplot(132)
plt.imshow(theta_est)
plt.title("theta estimated")
plt.subplot(133)
plt.plot(data)
plt.show()



#np.set_printoptions(precision=4)
#print("theta_est = \n", theta_est)
