#############################################################
#	generate n dimensional smpling that is that{(x1,x2,....,xn)}
#	in this case parameter theta is assume given
#       modification: sample valiables from the part of sequence. 
#       modification: using simulated anealign
#   ---> X[i,k], theta[k,i,j], i,j = index of spins, k = index of different temperture
##############################################################
import numpy as np
from scipy import linalg
np.random.seed(0)
################### set of prob parameters ###################
n = 8  # number of Ising variables
T = 25 # number of epoc for theta
N = 100 # number of smples, comes form true Prob
N_est = 100 # number of smples, comes form estimated Prob
ypc = 10 # descent ratio
cord = 0 # changiable spin variable when gibbus sampling time step
# set symmetric theta matrix
"anealing parameter"
beta = 1.0
K = 10 # number of differnt type respect to  beta
theta = np.arange(1,n+1)
theta = np.tensordot(theta[:, np.newaxis], theta[: , np.newaxis], axes = ([1],[1]))
np.fill_diagonal(theta, 0)
# labeling for theat 1 to K
a = np.ones(K)
# theta(i,j,k); k is different temperture
theta = np.tensordot(theta[np.newaxis,:,:], a[:,np.newaxis], axes=([0],[1]))
theta_est =  n * np.random.rand(n,n)    # create same size of matrix
theta_est = theta_est[np.newaxis,:,:]
theta_est = np.tensordot(theta_est, a[:,np.newaxis], axes=([0],[1]))
# set del theta mat
del_l_del_theta = np.empty_like(theta) # sum of x_i* x_j for each sample
del_l_del_theta_est = np.random.rand(n,n) # sum of x_i* x_j for each sample
del_l_del_theta_est = np.tensordot( del_l_del_theta_est[:,:,np.newaxis], a[:,np.newaxis] ,axes=([2],[1]) )

x =  2 * np.array(np.random.random_integers(0,1,n) - 0.5)
X = np.concatenate(([x],[x]), axis=0)
X_est =  np.concatenate(([x],[x]), axis=0)
for i in range(K-2):
    X = np.concatenate((X,[x]), axis=0)
    X_est =  np.concatenate(( X_est ,[x] ), axis=0)
# this func obtain one set of (x1,...,xn), which is distribute Boltzman waight
# t_wait_sample : waiting time to get sample, each time step is correspond to the update step on Gibbus sampling
def gen_a_gibbus_sample(t_wait_sample,X = [[]], theta=[[[]]]):
    # local minimamu が沢山ある分布なので、最初の状態でどれを選ぶかに大きく依存してしまう
    # Gibbus samplingはもちろん部分列を取り出さなければならない
    # sampling方法の修正、順番に分布関数を変化させていくが、samplingはランダムに部分列を取り出すことにする。
    # -> 変化可能な変数の選択をランダムにしてlocal minima にはまらないようにしよう
    global cord # this is selected 
    for i in range(t_wait_sample):# 20 = 収束するまでの更新回数
        cord = (cord + 1 ) % n 
        #valu = 0.1 * sum(  np.array( np.tensordot( X ,theta, axes = ([0],[1]) ) ) ) - X[cord] * theta[cord][cord]
        for k in range(K):
            valu = 0.01 * k *  ( np.dot( X[k,:], theta[:, cord, k]) -X[k,cord] * theta[cord,cord,k] ) 
            #print("valu = ", valu)
            r = np.exp( -valu ) / ( np.exp( -valu ) + np.exp( valu ) )
            R = np.random.uniform(0,1)
            if (R <= r):
                X[k,cord] =  1
            else:
                X[k,cord] = -1
    return X
# t_wait_sample 回数のupdate後のsampleが一つ手元に入ってく
# ここまででthetaが与えられたら、sampleを生成できるようになった。

# make a matrix (del_l_theta)_ij , Sum_k (x^k_i, x^k_j) matrix,k is sample index,  k = 1,...,N
# N_start, N_endの区間のみを使用するように
def get_del_l_del_theta_mat(N, X = [[]] , theta = [[[]]], del_l_del_theta = [[[]]]):
    for k in range(N):
        rand = np.random.randint(5,n * 10) # mean of rand is 20, which is same with previous one
        X = gen_a_gibbus_sample(rand, X , theta)
        for j in range(K):
            del_l_del_theta[:,:,j] = del_l_del_theta[:,:,j] + np.tensordot(X[j,:,np.newaxis],X[j,:,np.newaxis], axes = ([1],[1]))
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
    for k in range(K):
        print( (np.absolute( theta[:,:,k] - theta_est[:,:,k] ).sum() ) , " ")
    print("")
print("theta_est = \n", np.absolute(theta_est-theta))
