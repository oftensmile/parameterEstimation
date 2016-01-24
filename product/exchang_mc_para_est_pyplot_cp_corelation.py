##############################################################
#   get samples through many local minimum points
#   sampling from enough time distance -> sampling時間が短く相関が強いために誤差関数は単調増加だった
#   check sampled datas are actually sample from target pdf using analysis corelation of tow spin
##############################################################
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import time 
####################### set parameters ########################
"""" Generate sampring by one MC-step for different beta """
""" Can I use sampe Theta-Matrix for differnet replicas ? """
time_start = time.time()
np.random.seed(0)
def calc_steady_prob(index_spin, x = [], theta = [[]]):
    a =  -0.01 * (  x[index_spin] * np.tensordot(x, theta[index_spin, :], axes = ([0],[0])) -  theta[index_spin, index_spin] * x[index_spin] )
    return np.exp(a)
        
def beta_power_of_prob(index_spin,  beta, x = [], theta = [[]]):
    p_of_x = calc_steady_prob(index_spin, x, theta) 
    if(p_of_x != 0):
        a =  ( p_of_x ** beta )/(  p_of_x ** beta  +  p_of_x ** (-beta) )
    else:
        a = 0
    return a

def gibbus_algorithm(beta,index_spin, index_replica, x = [], theta = [[]]):
    x_proposed = x
    x_proposed[index_spin] *= -1
    if( np.random.uniform(size=1) < beta_power_of_prob(index_spin, beta, x_proposed, theta)  ):
        return x_proposed
    else:
        return x

def gen_sample_from_gibbus(index_spin, num_spin, num_replica, beta_min, d_beta, X = [[]], theta = [[]]):
    for index_replica in range(num_replica):
        beta = beta_min + index_replica * d_beta
        X[index_replica,:] = gibbus_algorithm(beta, index_spin, index_replica, X[index_replica,:] , theta)
        #print("X=\n", X)
    return X

def replica_exchange(index_spin, index_replica, beta, d_beta,X = [[]] ):
    x1, x2 = X[index_replica, :], X[index_replica + 1, :]
    # r = P(x_k | beta_k+1)P(x_k+1 | beta_k) / P(x_k | beta_k)P(x_k+1 | beta_k+1)
    r = beta_power_of_prob(index_spin, beta + d_beta, x1, theta) * beta_power_of_prob(index_spin, beta, x2, theta) / beta_power_of_prob(index_spin, beta, x1, theta) * beta_power_of_prob(index_spin, beta +d_beta, x2, theta)
    if(np.random.uniform(size=1) < r):
        X[index_replica, :], X[index_replica + 1, :] = np.copy(x2), np.copy(x1)
    return X

def sum_of_xi_xj_over_num_sample(x_mc_sequence = []):
    num_sample = len(x_mc_sequence)
    return np.tensordot(x_mc_sequence, x_mc_sequence, axes = ([0],[0])) / float(len (x_mc_sequence))

####################################   M A I N   ########################################
if __name__ == "__main__":
    num_spin, num_replica = 8, 3 # nuber of spin variable and replica
    beta_min, beta_max = 0.01, 1 # inverse temp of min, max
    d_beta = float(beta_max- beta_min) / num_replica
    exchange_interval = 3 # number of mc-steps between exchangings of replica index
    sampling_interval = 9
    num_sample, num_sample_est, num_corelation = 5000, 100, 1000 # number of sampling from true, estimated  prob
    #theta = np.arange(1, num_spin + 1)
    #theta = np.tensordot(theta[:, np.newaxis], theta[: , np.newaxis], axes = ([1],[1]))
    #np.fill_diagonal(theta, 0)
    #theta *= 0.01
    theta = [[0,1,0,0,0,0,0,1],
             [1,0,1,0,0,0,0,0],
             [0,1,0,1,0,0,0,0],
             [0,0,1,0,1,0,0,0],
             [0,0,0,1,0,1,0,0],
             [0,0,0,0,1,0,1,0],
             [0,0,0,0,0,1,0,1],
             [1,0,0,0,0,0,1,0]]
    theta = np.array(theta)
    theta_est = np.random.rand(num_spin,num_spin)    # create same size of matrix
    x =  2 * np.array(np.random.random_integers(0,1,num_spin) - 0.5)
    X , X_est = np.concatenate(([x],[x]), axis=0), np.concatenate(([x],[x]), axis=0)
    for i in range(num_replica -2):
        X, X_est = np.concatenate((X,[x]), axis=0), np.concatenate((X_est,[x]), axis=0)
    #record sequence of MC-sampling for lagest beta(lowest temperature)
    epoc_theta = 100 # number of epoc
    ypc = 0.1   #  C R U C I A L   #
    #for estimation of parameter theata-atrix, sum of xi * xj

    # sampling from true probability
    index_spin = 0
    mean_xxT = np.zeros((num_spin, num_spin))
    for t in range(num_sample): ######check point######gibbus algorithm自身は毎回していなければ意味がないのではないだろうか？
        index_spin = (t) % num_spin
        index_replica =  np.random.randint(0, num_replica - 1 )
        for i in range(num_replica):
            beta = beta_min + d_beta * i
            X[i] = gibbus_algorithm(beta,index_spin, i, X[i], theta)
        if(t % sampling_interval == 0):########check ########index_spin が変化せず０のままではないか？
            X = gen_sample_from_gibbus(index_spin, num_spin, num_replica, beta_min,d_beta, X, theta)
            if(t % exchange_interval == 0):
                #####check point####2n+*1 番目のspin_indexを選んで(n=0,1,..)それを隣接する変数と交換する方が良い
                beta = beta_min + d_beta * index_replica
                X = replica_exchange(index_spin, index_replica, beta, d_beta, X)
            x_L_beta = np.array( X[num_replica-1, :] )  #it must be done for all replicas
            mean_xxT = mean_xxT + np.tensordot(x_L_beta[:,np.newaxis], x_L_beta[:,np.newaxis], axes = ([1],[1]))
        mean_xxT = ( 1.0 / num_sample) * mean_xxT
    
    # estimation of theta mat
    data = np.zeros(epoc_theta)
    for epoc in range(epoc_theta):
        # reset
        x =  2 * np.array(np.random.random_integers(0,1,num_spin) - 0.5)
        X_est = np.concatenate(([x],[x]), axis=0)
        for i in range(num_replica -2):
            X_est = np.concatenate((X_est,[x]), axis=0)
                
        # sampling from estimated probability
        mean_xxT_est = np.zeros((num_spin, num_spin))
        for t in range(num_sample_est):
            index_spin = (t) % num_spin
            if(t % sampling_interval == 0):
                index_spin = 0
                X_est = gen_sample_from_gibbus(index_spin, num_spin, num_replica, beta_min,d_beta, X_est, theta_est)
                if(t % exchange_interval == 0):
                    index_replica =  np.random.randint(0, num_replica - 1 ) 
                    beta = beta_min + d_beta * index_replica
                    X_est = replica_exchange(index_spin, index_replica, beta, d_beta, X_est)
                x_L_beta = np.array( X_est[num_replica-1, :]  ) 
                A = np.tensordot(x_L_beta[:,np.newaxis], x_L_beta[:,np.newaxis], axes = ([1],[1]))
                mean_xxT_est = mean_xxT_est + A #np.tensordot(x_L_beta[:,np.newaxis], x_L_beta[:,np.newaxis], axes = ([1],[1]))
            mean_xxT_est = ( 1.0 / num_sample_est) * mean_xxT_est
            theta_est = theta_est - ypc * (-1) * ( mean_xxT - mean_xxT_est ) # (-1) correspond to (- beta ) 
        data[epoc] = np.absolute( theta - theta_est ).sum()
        #print("theta_est = \n", theta_est)
        print(data[epoc])
    time_end = time.time()
    print("#running time = ", time_end - time_start)
    # make sure for the corelation function
    corelation = 0.0
    x_corelation = 2 * np.array(np.random.random_integers(0,1,num_spin) - 0.5)
    X_corelation = np.concatenate(([x_corelation],[x_corelation]), axis=0)
    for i in range(num_replica -2):
        X_corelation = np.concatenate((X_corelation,[x_corelation]), axis=0)
    for j in range(num_corelation * num_spin):
        if(j % (5 * num_spin) == 0):
            X_corelation = gen_sample_from_gibbus( j % num_spin,num_spin, num_replica, beta_min, d_beta, X_corelation, theta )
            x_corelation = X_corelation[num_replica-1]
            print("x_corelation = ", x_corelation)
            corelation += x_corelation[1] * x_corelation[2]
    corelation /= (num_corelation / 5 )
    print("corelation = ", corelation)
    th8 = np.tanh(-1)
    corelation_theory = ( th8 + th8 ** 7 )/ (1 + th8 ** 8)
    print("corelation_theory = ", corelation_theory)

    plt.plot(data)
    plt.ylabel('Error Function', fontsize='20')
    plt.xlabel('theta update-step', fontsize='20')
    plt.show()
    
    # maybe above process created estimated theta-matrix 

