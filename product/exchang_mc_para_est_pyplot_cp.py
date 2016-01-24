##############################################################
#   get samples through many local minimum points
#   sampling from enough time distance -> sampling時間が短く相関が強いために誤差関数は単調増加だった
#   modified after individual meeting
##############################################################
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import time
####################### set parameters ########################
"""" Generate sampring by one MC-step for different beta """
""" Can I use sampe Theta-Matrix for differnet replicas ? """
start_time = time.time()
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
    return X

def replica_exchange(index_spin, index_replica, beta, d_beta,X = [[]] ):
    x1, x2 = X[index_replica, :], X[index_replica + 1, :]
    # r = P(x_k | beta_k+1)P(x_k+1 | beta_k) / P(x_k | beta_k)P(x_k+1 | beta_k+1)
    r = beta_power_of_prob(index_spin, beta + d_beta, x1, theta) * beta_power_of_prob(index_spin, beta, x2, theta) / beta_power_of_prob(index_spin, beta, x1, theta) * beta_power_of_prob(index_spin, beta +d_beta, x2, theta)
    if(np.random.uniform(size=1) < r):
        X[index_replica, :], X[index_replica, :] = np.copy(x2), np.copy(x1)
    return X

def sum_of_xi_xj_over_num_sample(x_mc_sequence = []):
    num_sample = len(x_mc_sequence)
    return np.tensordot(x_mc_sequence, x_mc_sequence, axes = ([0],[0])) / float(len (x_mc_sequence))

####################################   M A I N   ########################################
if __name__ == "__main__":
    num_spin, num_replica = 8, 3 # nuber of spin variable and replica
    beta_min, beta_max = 0.001, 0.01 # inverse temp of min, max
    d_beta = float(beta_max- beta_min) / num_replica
    num_sample, num_sample_est = 1000, 100 # number of sampling from true, estimated  prob
    epoc_theta = 100 # number of epoc
    ypc = 0.1   #  C R U C I A L   #
    sampling_interval = 5
    exchange_interval = 5 # number of mc-steps between exchangings of replica index

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
    theta_est_tr = np.transpose(theta_est)
    theta_est = theta_est + theta_est_tr
    np.fill_diagonal(theta_est, 0)
    print("theta_est = \n", theta_est)
    x =  2 * np.array(np.random.random_integers(0,1,num_spin) - 0.5)
    X , X_est = np.concatenate(([x],[x]), axis=0), np.concatenate(([x],[x]), axis=0)
    for i in range(num_replica -2):
        X, X_est = np.concatenate((X,[x]), axis=0), np.concatenate((X_est,[x]), axis=0)
    #record sequence of MC-sampling for lagest beta(lowest temperature)
    #for estimation of parameter theata-atrix, sum of xi * xj

    # sampling from true probability
    index_spin = 0
    mean_xxT = np.zeros((num_spin, num_spin))
    for t in range(1,num_sample * num_spin * sampling_interval):
        if(t % (sampling_interval * num_spin)== 0):
            X = gen_sample_from_gibbus(index_spin, num_spin, num_replica, beta_min,d_beta, X, theta)
            if(t % (exchange_interval * num_spin) == 0):
                index_replica =  np.random.randint(0, num_replica - 1 )
                beta = beta_min + d_beta * index_replica
                X = replica_exchange(index_spin, index_replica, beta, d_beta, X)
            x_L_beta = np.array( X[num_replica-1, :] )  #it must be done for all replicas
            mean_xxT = mean_xxT + np.tensordot(x_L_beta[:,np.newaxis], x_L_beta[:,np.newaxis], axes = ([1],[1]))
            index_spin = (t) % num_spin
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
        for t in range(1,num_sample_est * num_spin * sampling_interval):
            if(t % (exchange_interval * num_spin) == 0):
                index_replica =  np.random.randint(0, num_replica - 1 ) 
                beta = beta_min + d_beta * index_replica
                print("before exchange \n", X_est)
                X_est = replica_exchange(index_spin, index_replica, beta, d_beta, X_est)
                print("after exchange \n", X_est)
            if(t % (sampling_interval * num_spin) == 0):
                #Should I initialize X after this sampling ?
                index_spin = (t) % num_spin
                print("before sampling \n", X_est)
                X_est = gen_sample_from_gibbus(index_spin, num_spin, num_replica, beta_min,d_beta, X_est, theta_est)
                print("after sampling \n", X_est)
            x_L_beta = np.array( X_est[num_replica-1, :]  )
            A = np.tensordot(x_L_beta[:,np.newaxis], x_L_beta[:,np.newaxis], axes = ([1],[1]))
            mean_xxT_est = mean_xxT_est + A #np.tensordot(x_L_beta[:,np.newaxis], x_L_beta[:,np.newaxis], axes = ([1],[1]))
        mean_xxT_est = ( 1.0 / num_sample_est) * mean_xxT_est
        theta_est = theta_est - ypc * (-1) * ( mean_xxT - mean_xxT_est ) # (-1) correspond to (- beta ) 
        data[epoc] = np.absolute( theta - theta_est ).sum()
        print(data[epoc])
    print("#running_time = %s" % (time.time() - start_time))
    plt.plot(data)
    plt.ylabel('Error Function', fontsize='20')
    plt.xlabel('theta update-step', fontsize='20')
    plt.show()
    # maybe above process created estimated theta-matrix 
