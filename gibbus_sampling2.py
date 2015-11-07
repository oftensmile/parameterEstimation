#############################################################
#	generate n dimensional smpling that is that{(x1,x2,....,xn)}
#	in this case parameter theta is assume given
##############################################################
import numpy as np
from scipy import linalg
np.random.seed(0)
################### set of prob parameters ###################
n = 8  # number of Ising variables
T = 400 # number of epoc for theta
N = 10000 # number of smples, comes form true Prob
N_est = 100 # number of smples, comes form estimated Prob
ypc = 10 # descent ratio
# set symmetric theta matrix
theta = np.arange(1,n+1)
theta = np.tensordot(theta[:, np.newaxis], theta[: , np.newaxis], axes = ([1],[1]))
np.fill_diagonal(theta, 0)
print("theta = \n", theta)
theta_est =  n * np.random.rand(n,n)    # create same size of matrix
print("theta_est = \n", theta_est)
# set del theta mat
del_l_del_theta = np.empty_like(theta) # sum of x_i* x_j for each sample
del_l_del_theta_est = np.random.rand(n,n) # sum of x_i* x_j for each sample
#X = np.ones(n)
#X_est = np.ones(n)

X =  2 * np.array(np.random.random_integers(0,1,n) - 0.5)
X_est =  2 * np.array(np.random.random_integers(0,1,n) - 0.5)

# this func obtain one set of (x1,...,xn), which is distribute Boltzman waight
def gen_a_gibbus_sample(X = [], theta=[[]]):
    # local minimamu が沢山ある分布なので、最初の状態でどれを選ぶかに大きく依存してしまう
    # 解決方法としては初期状態をランダムに幾つかセットして定常状態に落ち着かせ、
    # １sampleだけ取り出して、また新しい初期状態にする。
    # -> 変化可能な変数の選択をランダムにしてlocal minima にはまらないようにしよう
    for i in range(20):# 20 = 収束するまでの更新回数
        cord = np.random.randint(n) # select flip int value
        #valu = 0.1 * sum(  np.array( np.tensordot( X ,theta, axes = ([0],[1]) ) ) ) - X[cord] * theta[cord][cord] 
        valu = 0.1 *  ( np.tensordot(X, theta[:,cord], axes = ([0],[0]) ) -X[cord] * theta[cord][cord] ) 
        r = np.exp( -valu ) / ( np.exp( -valu ) + np.exp( valu ) )
        R = np.random.uniform(0,1)
        if (R <= r):
            X[cord] =  1
        else:
            X[cord] = -1
    return X
# ここまででthetaが与えられたら、sampleを生成できるようになった。

# make a matrix (del_l_theta)_ij , Sum_k (x^k_i, x^k_j) matrix,k is sample index,  k = 1,...,N
# N_start, N_endの区間のみを使用するように
def get_del_l_del_theta_mat(N, X = [] , theta = [[]], del_l_del_theta = [[]]):
    for k in range(N):
        X = gen_a_gibbus_sample(X , theta)
        del_l_del_theta = del_l_del_theta + np.tensordot(X[:,np.newaxis],X[:,np.newaxis], axes = ([1],[1]))
    return del_l_del_theta / N
# ここまでで勾配をもとめる各thetaのij微分がもとまるようになった。
############################ MAIN ###############################
del_l_del_theta = get_del_l_del_theta_mat(N, X,theta, del_l_del_theta)
# 以下では1sampleでdel_l_del_theta_matを求めて、thetaを更新してを繰り返す。
for t in range(T):
    # update theta using Graduant Descent
    theta_est = theta_est -  ypc * ( del_l_del_theta - del_l_del_theta_est )
    #theta_est = theta_est -  ypc * ( - del_l_del_theta_est )
    # sampling using t-th theta
    del_l_del_theta_est = get_del_l_del_theta_mat(N_est, X_est, theta_est, del_l_del_theta_est)
    print( (np.absolute( theta - theta_est ).sum() ))
print("theta_est = \n", theta_est)
