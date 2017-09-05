#############################################################
#
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
T = 200 # number of epoc for theta
N = 1000 # number of smples, comes from true Prob
N_est = 20 # number of smples, come from estimated Prob
ypc = 1 # descent ratio
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
X =  2 * np.array(np.random.random_integers(0,1,n) - 0.5)
X_est =  2 * np.array(np.random.random_integers(0,1,n) - 0.5)
# this func obtain one set of (x1,...,xn), which is distribute Boltzman waight
# t_wait_sample : waiting time to get sample, each time step is correspond to the update step on Gibbus sampling
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
    del_l_del_theta_est = get_del_l_del_theta_mat(N_est, X_est, theta_est)
    theta_est = theta_est -  ypc * ( del_l_del_theta - del_l_del_theta_est )
    data[t] = np.absolute( theta - theta_est ).sum()
    #print( data[t] )
time_end = time.time()
#   estimateされたthetaを使ってxのサンプル列を得る
N_errorbar = 10
x_errorbar = 2 * np.array(np.random.random_integers(0,1,n) - 0.5)
x_errorbar =  gen_a_gibbus_sample(10*n, x_errorbar, theta_est)
theta_var_12,theta_var_12_mean = 0.0 ,0.0

N_error_bar = 10
for k in range(N_error_bar):
    #[提案分布]   p = const. exp(-|| theta - theta_est ||**2)に従って thetaをサンプルする。ただし変更するthetaはひとつのみ
    M = 100 # 1つのsample x_nから作るtheta_smapleの数
    for m in range(M):
        theta_sample = np.zeros((n,n),dtype=np.float) # initial state
        #   この更新の方法ではひとつ前の更新時のtheta_sampleが使われていない、毎回新しくsampleを作っているため収束しようがない
        for t in range(100):# theta_sampleを一巡する回数
            for i in range(n):
                for j in range(i+1,n):
                    del_theta = float(np.random.randn(1)) # = theta_sample - theta_est
                    theta_sample_ij = del_theta + theta_est[i][j] # mean=thea_est[i,j] 's gauss ditribution
                    wait_ratio = np.exp(- del_theta * x_errorbar[i]*x_errorbar[j])
                    Q_ratio = np.exp( - 0.5 *del_theta**2 + del_theta ) # Q := exp(- 0.5 *|| theta_sample - theta_est||**2) 
                    if ( wait_ratio > 1 ):
                        r, R = wait_ratio / Q_ratio, np.random.uniform(0,1)
                        if(r >= R):
                            theta_sample[i,j] = del_theta + theta_est[i,j]
                            theta_sample[j,i] = theta_sample[i,j]
        theta_var_12 += (theta_sample[1,2] - theta_est[1,2] )**2
    theta_var_12 /= M
    theta_var_12_mean += theta_var_12
    #print("#thata_var_12(k) = ", theta_var_12)
theta_var_12_mean /= N_error_bar
print("theta_var_12_mean = ",theta_var_12_mean)
#   あるサンプルx_nでthetaが収束したらそこからx_nに依存するthetaを多数samplingしthetaに対しての分散を作る

#連続変数のthetaはmcmcする必要があるのだろうか？








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
