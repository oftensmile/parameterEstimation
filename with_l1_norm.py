#############################################################
#	generate n dimensional smpling that is that{(x1,x2,....,xn)}
#	in this case parameter theta is assume given
#       modification: sample valiables from the part of sequence. 
#       with L1-regularization, i.e, del_|theta|_ij = sign(theta_ij)
##############################################################
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import time
import sys
################### set of prob parameters ###################
time_start = time.time()
np.random.seed(0)
n = 8  # number of Ising variables
T = 800 # number of epoc for theta
N = 1000 # number of smples, comes from true Prob
N_est = 20 # number of smples, come from estimated Prob
ypc = 1 # descent ratio
alpha = 0.1
N_corelation = 1000
# set symmetric theta matrix
#theta = np.arange(1,n+1)
#theta = np.tensordot(theta[:, np.newaxis], theta[: , np.newaxis], axes = ([1],[1]))
#np.fill_diagonal(theta, 0)
theta_est =  np.random.rand(n,n)    # create same size of matrix
np.fill_diagonal(theta_est, 0)
theta_tr = np.transpose(theta_est)
theta_est = theta_est + theta_tr
np.fill_diagonal(theta_est, 0)
theta =[[0,1,0,0,0,0,0,1],
        [1,0,1,0,0,0,0,0],
        [0,1,0,1,0,0,0,0],
        [0,0,1,0,1,0,0,0],
        [0,0,0,1,0,1,0,0],
        [0,0,0,0,1,0,1,0],
        [0,0,0,0,0,1,0,1],
        [1,0,0,0,0,0,1,0]]

theta = np.array(theta)

# set del theta mat
del_l_del_theta = np.empty_like(theta) # sum of x_i* x_j for each sample
del_l_del_theta_est = np.empty_like(theta) # sum of x_i* x_j for each sample
#X = np.ones(n)
#X_est = np.ones(n)

X =  2 * np.array(np.random.random_integers(0,1,n) - 0.5)
X_est =  2 * np.array(np.random.random_integers(0,1,n) - 0.5)
# this func obtain one set of (x1,...,xn), which is distribute Boltzman waight
# t_wait_sample : waiting time to get sample, each time step is correspond to the update step on Gibbus sampling
def gen_a_gibbus_sample(t_wait_sample,X = [], theta=[[]]):
    for i in range(t_wait_sample * n):# 20 = 収束するまでの更新回数
        cord = i  % n 
        #print("twait = ", t_wait_sample, " cord = ", cord)
        #cord = np.random.randint(n) # select f
        #valu = 0.1 * sum(  np.array( np.tensordot( X ,theta, axes = ([0],[1]) ) ) ) - X[cord] * theta[cord][cord] 
        valu =   ( np.tensordot(X, theta[:,cord], axes = ([0],[0]) ) -X[cord] * theta[cord][cord] ) 
        r = np.exp( -valu ) / ( np.exp( -valu ) + np.exp( valu ) )
        R = np.random.uniform(0,1)
        if (R <= r):
            X[cord] =  1
        else:
            X[cord] = -1
    return X
# t_wait_sample 回数のupdate後のsampleが一つ手元に入ってく
# ここまででthetaが与えられたら、sampleを生成できるようになった。

# make a matrix (del_l_theta)_ij , Sum_k (x^k_i, x^k_j) matrix,k is sample index,  k = 1,...,N
# N_start, N_endの区間のみを使用するように
def get_del_l_del_theta_mat(N, X = [] , theta = [[]]):
    del_l_del_theta = np.zeros((n,n), dtype=np.float)
    for k in range(N):
        X = gen_a_gibbus_sample(10, X , theta)
        del_l_del_theta = del_l_del_theta + np.tensordot(X[:,np.newaxis],X[:,np.newaxis], axes = ([1],[1]))
    return del_l_del_theta / N
# ここまでで勾配をもとめる各thetaのij微分がもとまるようになった。
############################ MAIN ###############################
del_l_del_theta = get_del_l_del_theta_mat(N, X,theta)
# 以下では1sampleでdel_l_del_theta_matを求めて、thetaを更新してを繰り返す。
data = np.zeros(T)
for t in range(T):
    ypc = 1.0 /np.log(t + 2)
    # update theta using Graduant Descent
    #theta_est = theta_est -  ypc * ( - del_l_del_theta_est )
    # sampling using t-th theta
    del_l_del_theta_est = get_del_l_del_theta_mat(N_est, X_est, theta_est)
    # with L1-regularization 
    theta_est = theta_est -  ypc * ( del_l_del_theta - del_l_del_theta_est + alpha * np.sign(theta_est) )
    data[t] = np.absolute( theta - theta_est ).sum()
    print( data[t] )
corelation = 0
corelation_est = 0
time_end = time.time()
for j in range(N_corelation * 10):
    x_for_corelation =  2 * np.array(np.random.random_integers(0,1,n) - 0.5)
    x_for_corelation_est =  2 * np.array(np.random.random_integers(0,1,n) - 0.5)
    x_for_corelation = gen_a_gibbus_sample(10*n, x_for_corelation, theta)
    x_for_corelation_est = gen_a_gibbus_sample(10*n, x_for_corelation_est, theta_est)
    corelation += x_for_corelation[1] * x_for_corelation[2]
    corelation_est += x_for_corelation_est[1] * x_for_corelation_est[2]
corelation = corelation / N
corelation_est = corelation_est / N

plt.plot(data)
plt.show()

with open("matrix.txt",  "w") as file:
    file.write("theta = \n")
    for i in range(n):
        file.write("#" + str(theta[i]) + "\n")
    file.write("theta = \n")
    for i in range(n):
        file.write("#" + str( theta_est[i]) + "\n")

print("#running time = ", time_end - time_start)
print("#corelation = ", corelation)
print("#corelation_est = ", corelation_est)
th8 = np.tanh(-1)
corelation_theory = ( th8 + th8 ** 7 )/ (1 + th8 ** 8)
print("#corelation_theory = ", corelation_theory)
