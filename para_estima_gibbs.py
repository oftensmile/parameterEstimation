##############################################################
#   get samples through many local minimum points
#
##############################################################
import numpy as np
from scipy import linalg
np.random.seed(0)
####################### set parameters ########################
"""" Generate sampring by one MC-step for different beta """
""" Can I use sampe Theta-Matrix for differnet replicas ? """
def calc_steady_prob(index_spin, x = [], theta = [[]]):
    return np.exp( -  x[index_spin] * np.tensordot(x, theta[index_spin, :], axes = ([0],[0])) -  theta[index_spin, index_spin] * x[index_spin]  )

def beta_power_of_prob(index_spin,  beta, x = [], theta = [[]]):
    p_of_x = calc_steady_prob(index_spin, x, theta)
    return (- p_of_x **  beta )/(  p_of_x ** beta  +  p_of_x ** (-beta) )

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

def replica_exchange(index_spin, index_replica,beta, d_beta,X = [[]] ):
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
    num_spin, num_replica = 6, 5 # nuber of spin variable and replica
    beta_min, beta_max = 0.005, 0.01 # inverse temp of min, max
    d_beta = float(beta_max- beta_min) / num_replica
    exchange_interval = 20 # number of mc-steps between exchangings of replica index
    num_sample, num_sample_est = 1000, 100 # number of sampling from true, estimated  prob

    theta = np.arange(1, num_spin + 1)
    theta = np.tensordot(theta[:, np.newaxis], theta[: , np.newaxis], axes = ([1],[1]))
    np.fill_diagonal(theta, 0)
    theta_est = np.random.rand(num_spin,num_spin)    # create same size of matrix
    x =  2 * np.array(np.random.random_integers(0,1,num_spin) - 0.5)
    X , X_est = np.concatenate(([x],[x]), axis=0), np.concatenate(([x],[x]), axis=0)
    for i in range(num_replica -2):
        X, X_est = np.concatenate((X,[x]), axis=0), np.concatenate((X,[x]), axis=0)
    #record sequence of MC-sampling for lagest beta(lowest temperature)
    epoc_theta = 300 # number of epoc
    ypc = 1.0   #  C R U C I A L   #
    #for estimation of parameter theata-atrix, sum of xi * xj
    mean_xxT, mean_xxT_est = np.empty_like(theta) , np.random.rand(num_spin, num_spin)

    # sampling from true probability
    for t in range(num_sample):
        index_spin = 0
        index_spin = (index_spin + 1) % num_spin
        gen_sample_from_gibbus(index_spin, num_spin, num_replica, beta_min,d_beta, X, theta)
        if(t % exchange_interval == 0):
            index_replica =  np.random.randint(0, num_replica - 1 ) 
            beta = beta_min + d_beta * index_replica
            replica_exchange(index_spin, index_replica, beta, d_beta, X)
        x_L_beta = np.array( X[num_replica-1, :] ) /  np.sqrt(num_sample) #only largest beta
        mean_xxT = mean_xxT + np.tensordot(x_L_beta[:,np.newaxis], x_L_beta[:,np.newaxis], axes = ([1],[1]))
    
    # estimation of theta mat
    for epoc in range(epoc_theta):
        # sampling from estimated probability
        for t in range(num_sample_est):
            index_spin = 0
            index_spin = (index_spin + 1) % num_spin
            gen_sample_from_gibbus(index_spin, num_spin, num_replica, beta_min,d_beta, X_est, theta_est)
            if(t % exchange_interval == 0):
                index_replica =  np.random.randint(0, num_replica - 1 ) 
                beta = beta_min + d_beta * index_replica
                replica_exchange(index_spin, index_replica, beta, d_beta, X_est)
            x_L_beta = np.array( X_est[num_replica-1, :] ) /  np.sqrt(num_sample_est) 
            mean_xxT_est = mean_xxT_est + np.tensordot(x_L_beta[:,np.newaxis], x_L_beta[:,np.newaxis], axes = ([1],[1]))
        #update of tehta-mat by SGD
        theta_est = theta_est - ypc * ( mean_xxT - mean_xxT_est )
        print( (np.absolute( theta - theta_est ).sum() ))
        # reset
        mean_xxT_est = np.empty_like(mean_xxT_est)
        x =  2 * np.array(np.random.random_integers(0,1,num_spin) - 0.5)
        X_est = np.concatenate(([x],[x]), axis=0)
        for i in range(num_replica -2):
            X_est = np.concatenate((X,[x]), axis=0)
    
    # maybe above process created estimated theta-matrix 


