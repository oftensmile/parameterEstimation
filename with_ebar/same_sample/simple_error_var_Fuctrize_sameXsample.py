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
T = 500 # number of epoc for theta
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
    # X_estを以降の分散を求める際に使用する
    if t==0:
        X_snap = np.vstack((X_est, X_est))
    else:
        X_snap = np.vstack((X_snap, X_est))
    
    theta_est = theta_est -  ypc * ( del_l_del_theta - del_l_del_theta_est )
    data[t] = np.absolute( theta - theta_est ).sum()
    print( data[t] )
time_end = time.time()
interval1 = time_end - time_start 
#   estimateされたthetaを使ってxのサンプル列を得る
time_start = time.time()
N_theta_sample = T  #1つのsample x_nから作るtheta_smapleの数
N_sample_snap = 40# theta error bar に使用するx sampleの数
N_sample_heat = 3 # sample for calclation of partion function under P(x|theta_prop)
T_theta_mcmc_interval = 10
#   この更新の方法ではひとつ前の更新時のtheta_sampleが使われていない、毎回新しくsampleを作っているため収束しようがない
theta_12_record = []
theta_mean_12 = 0.0
for t in range(N_theta_sample):
    theta_sample = np.random.uniform(0,1,(n,n)) # initial state
    theta_sample = theta_sample.T + theta_sample
    np.fill_diagonal(theta_sample,0)
    for tm in range(T_theta_mcmc_interval):# "no sampling running" of Theta
        for i in range(n):
            for j in range(i+1,n):
                # sample from steady state prob, which P(x|theta_prop) ~ x
                del_theta = float(np.random.randn(1)) 
                theta_sample_prop = theta_sample
                theta_sample_prop[i][j] += del_theta
                theta_sample_prop[j][i] += del_theta
                a = 0
                for m in range(N_sample_heat): 
                    x_heat = 2 * np.array(np.random.random_integers(0,1,n) - 0.5)
                    x_heat = gen_a_gibbus_sample(10*n, x_heat, theta_sample_prop)
                    a += np.exp(  x_heat[i] * x_heat[j] * del_theta) 
                a /= N_sample_heat
                b = del_theta * np.dot(X_snap[:,i],X_snap[:,j]) # = Sum_n ( x_i^(n)x_j^(n) )
                b = np.exp( - b )
                a = a**N_sample_snap * b

                r,R = a, np.random.uniform(0,1)
                if(r >= R):
                    theta_sample = theta_sample_prop
    

#ここまででthetaの更新が済んだと思う, mean, varianceを求める
    theta_12_record = np.append(theta_12_record, theta_sample[1][2])
    theta_mean_12 += theta_sample[1][2]
theta_mean_12 /= N_theta_sample

theta_var_12 = 0.0
for t in range(N_theta_sample):
    theta_var_12 += (theta_12_record[t] - theta_mean_12) **2
theta_var_12 /= N_theta_sample
print("#theta_12_mean = ", theta_mean_12,"\n#theta_var_12 = ", theta_var_12)    
time_end = time.time()
print("#Running time of obtaining errorbar = ", time_end - time_start)
#file_mean.write("true:theta_est[1,2]="+str(theta_est[1,2]))
#file_mean.close()
#file_var.close()
#   あるサンプルx_nでthetaが収束したらそこからx_nに依存するthetaを多数samplingしthetaに対しての分散を作る

#連続変数のthetaはmcmcする必要があるのだろうか？

plt.plot(data)
plt.show()
with open("matrix.txt",  "w") as f:
    f.write("theta = \n")
    for i in range(n):
        f.write("#" + str(theta[i]) + "\n")
    f.write("theta = \n")
    for i in range(n):
        f.write("#" + str( theta_est[i]) + "\n")
f.close()
with open("mean_and_var_12.txt", "w") as f2:
    f2.write("#theta_mean_12 = " + str( theta_mean_12) )
    f2.write("\n#theta_var_12 = " + str(theta_var_12) )
f2.close()
