#############################################################
#	generate n dimensional smpling that is that{(x1,x2,....,xn)}
#	in this case parameter theta is assume given
#       modification: sample valiables from the part of sequence. 
##############################################################
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import time
################### set of prob parameters ###################
start_time = time.time()
np.random.seed(0)
n = 8  # number of Ising variables
T = 200 # number of epoc for theta
N = 1000 # number of smples, comes form true Prob
N_est = 100 # number of smples, comes form estimated Prob
ypc = 0.1 # descent ratio
# set symmetric theta matrix
theta =[[0,1,0,0,0,0,0,1],
        [1,0,1,0,0,0,0,0],
        [0,1,0,1,0,0,0,0],
        [0,0,1,0,1,0,0,0],
        [0,0,0,1,0,1,0,0],
        [0,0,0,0,1,0,1,0],
        [0,0,0,0,0,1,0,1],
        [1,0,0,0,0,0,1,0]]
theta = np.array(theta)
theta_est =  np.random.rand(n,n)    # create same size of matrix
theta_est_tr = np.transpose(theta_est)
theta_est = theta_est + theta_est_tr
np.fill_diagonal(theta_est, 0)
# set del theta mat
del_l_del_theta = np.empty_like(theta) # sum of x_i* x_j for each sample
del_l_del_theta_est = np.random.rand(n,n) # sum of x_i* x_j for each sample
#X = np.ones(n)
#X_est = np.ones(n)

X =  2 * np.array(np.random.random_integers(0,1,n) - 0.5)
X_est =  2 * np.array(np.random.random_integers(0,1,n) - 0.5)

# this func obtain one set of (x1,...,xn), which is distribute Boltzman waight
# t_wait_sample : waiting time to get sample, each time step is correspond to the update step on Gibbus sampling
def gen_a_gibbus_sample(t_wait_sample,X = [], theta=[[]]):
    global cord # this is selected 
    for i in range(t_wait_sample):# 20 = 収束するまでの更新回数
        cord = i  % n 
        #valu = 0.1 * sum(  np.array( np.tensordot( X ,theta, axes = ([0],[1]) ) ) ) - X[cord] * theta[cord][cord] 
        valu = ( np.tensordot(X, theta[:,cord], axes = ([0],[0]) ) -X[cord] * theta[cord][cord] ) 
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
    del_l_del_theta = np.zeros((n,n),dtype=np.float)
    for k in range(N):
        X = gen_a_gibbus_sample(20, X , theta)
        del_l_del_theta = del_l_del_theta + np.tensordot(X[:,np.newaxis],X[:,np.newaxis], axes = ([1],[1]))
    return del_l_del_theta / N
# ここまでで勾配をもとめる各thetaのij微分がもとまるようになった。
############################ MAIN ###############################
del_l_del_theta = get_del_l_del_theta_mat(N, X,theta)
# 以下では1sampleでdel_l_del_theta_matを求めて、thetaを更新してを繰り返す。
data = np.zeros(T)
for t in range(T):
    # update theta using Graduant Descent
    # sampling using t-th theta
    del_l_del_theta_est = get_del_l_del_theta_mat(N_est, X_est, theta_est)
    theta_est = theta_est -  ypc * ( del_l_del_theta - del_l_del_theta_est )
    data[t] = np.absolute( theta - theta_est ).sum()
    print(data[t])
print("#runnig_time = %s", (time.time() - start_time))
#print("theta_est = \n", theta_est)
plt.plot(data)
plt.ylabel('Error Function',fontsize='20')
plt.xlabel('theta update-step', fontsize='20')
plt.show()
